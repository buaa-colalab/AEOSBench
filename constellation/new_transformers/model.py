from asyncio import tasks
from typing import Any, Iterable
import einops
import todd
import torch
from torch import nn

from todd.models.modules.transformer import Block
from todd.models.modules import sinusoidal_position_embedding
from todd.patches.torch import Sequential
from todd.models.losses import CrossEntropyLoss, MSELoss
from todd.registries import InitWeightsMixin
from constellation.data import SensorType
from .dataset import Batch
from .registries import ConstellationModelRegistry
from .time_model import TimeModel
from todd.runners import Memo, BaseRunner
from todd.registries import InitWeightsMixin
from todd.runners.callbacks import TensorBoardCallback
from constellation import MAX_TIME_STEP
from torch.distributions import Categorical
from .constants import SATELLITE_DIM, TASK_DIM

GLOBALS = dict()


class Encoder(nn.Module):

    def __init__(
        self,
        *args,
        time_embedding_dim: int,
        sensor_type_embedding_dim: int,
        data_dim: int = TASK_DIM,
        data_embedding_dim: int,
        width: int,
        depth: int,
        num_heads: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._num_heads = num_heads

        self._data_embedding = nn.Linear(data_dim, data_embedding_dim)
        self._in_projector = nn.Linear(
            time_embedding_dim + sensor_type_embedding_dim
            + data_embedding_dim,
            width,
        )
        self._blocks = Sequential(
            *[Block(width=width, num_heads=num_heads) for _ in range(depth)],
        )
        self._norm = nn.LayerNorm(width)

    def forward(
        self,
        time_embedding: torch.Tensor,
        sensor_type_embedding: torch.Tensor,
        data: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = einops.repeat(
            time_embedding,
            'b d -> b nt d',
            nt=data.shape[1],
        )
        data_embedding = self._data_embedding(data)
        embedding = torch.cat((
            time_embedding,
            sensor_type_embedding,
            data_embedding,
        ), -1)
        x = self._in_projector(embedding)
        attention_mask = (
            einops.rearrange(attention_mask, 'b nt -> b nt 1')
            & einops.rearrange(attention_mask, 'b nt -> b 1 nt')
        )
        attention_mask = einops.repeat(
            attention_mask,
            'b nt nt_prime -> (b nh) nt nt_prime',
            nh=self._num_heads,
        )
        attention_mask = torch.where(attention_mask, 0, float('-inf'))
        x = self._blocks(x, attention_mask=attention_mask)
        x = self._norm(x)
        return x


class DecoderBlock(Block):

    def __init__(
        self,
        *args,
        width: int,
        num_heads: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, width=width, num_heads=num_heads, **kwargs)
        self._norm3 = nn.LayerNorm(width, 1e-6)
        self._cross_attention = nn.MultiheadAttention(
            width,
            num_heads,
            batch_first=True,
        )

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        hidden_states: torch.Tensor,
        cross_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = super().forward(x, attention_mask)

        norm = self._norm3(x)
        cross_attention, _ = self._cross_attention(
            norm,
            hidden_states,
            hidden_states,
            need_weights=False,
            attn_mask=cross_attention_mask,
        )
        x = x + cross_attention

        return x


class Decoder(InitWeightsMixin, nn.Module):

    def __init__(
        self,
        *args,
        time_embedding_dim: int,
        sensor_type_embedding_dim: int,
        sensor_enabled_embedding_dim: int,
        data_dim: int = SATELLITE_DIM,
        data_embedding_dim: int,
        width: int,
        depth: int,
        num_heads: int,
        return_logits: bool,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._num_heads = num_heads

        self._sensor_enabled_embedding = nn.Embedding(
            2,
            sensor_enabled_embedding_dim,
        )
        self._data_embedding = nn.Linear(data_dim, data_embedding_dim)

        self._in_projector = nn.Linear(
            time_embedding_dim + sensor_type_embedding_dim
            + sensor_enabled_embedding_dim + data_embedding_dim,
            width,
        )

        self._blocks = Sequential(
            *[
                DecoderBlock(width=width, num_heads=num_heads)
                for _ in range(depth)
            ],
        )
        self._norm = nn.LayerNorm(width)

        if return_logits:
            self._null_task = nn.Parameter(torch.empty(width))

    @property
    def return_logits(self) -> bool:
        return hasattr(self, '_null_task')

    def init_weights(self, config: todd.Config) -> bool:
        if self.return_logits:
            self._null_task.data.zero_()
        return super().init_weights(config)

    def forward(
        self,
        time_embedding: torch.Tensor,
        sensor_type_embedding: torch.Tensor,
        sensor_enabled: torch.Tensor,
        data: torch.Tensor,
        mask: torch.Tensor,
        hidden_states: torch.Tensor,
        tasks_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        time_embedding = einops.repeat(
            time_embedding,
            'b d -> b ns d',
            ns=data.shape[1],
        )
        sensor_enabled_embedding = self._sensor_enabled_embedding(
            sensor_enabled,
        )
        data_embedding = self._data_embedding(data)
        embedding = torch.cat((
            time_embedding,
            sensor_type_embedding,
            sensor_enabled_embedding,
            data_embedding,
        ), -1)
        x = self._in_projector(embedding)

        mask = torch.where(mask, 0, float('-inf'))
        attention_mask = einops.repeat(
            mask,
            'b ns -> (b nh) ns ns_prime',
            nh=self._num_heads,
            ns_prime=embedding.shape[1],
        )
        cross_attention_mask = einops.repeat(
            tasks_mask,
            'b nt -> b ns nt',
            ns=data.shape[1],
        )
        cross_attention_mask = torch.where(
            cross_attention_mask,
            time_mask,
            # 0,
            float('-inf'),
        )
        cross_attention_mask = einops.repeat(
            cross_attention_mask,
            'b ns nt -> (b nh) ns nt',
            nh=self._num_heads,
        )

        x = self._blocks(
            x,
            attention_mask=attention_mask,
            hidden_states=hidden_states,
            cross_attention_mask=cross_attention_mask,
        )
        x = self._norm(x)

        if not self.return_logits:
            return x

        null_logits = torch.einsum('b s d, d -> b s', x, self._null_task)

        logits_mask = einops.rearrange(tasks_mask, 'b nt -> b 1 nt')
        logits = torch.einsum('b s d, b t d -> b s t', x, hidden_states)
        logits = logits + logits_mask

        return null_logits, logits
        # x = self._out_projector(x)
        # return x


class Transformer(nn.Module):

    def __init__(
        self,
        *args,
        time_embedding_dim: int = 64,
        sensor_type_embedding_dim: int,
        tasks_data_embedding_dim: int,
        encoder_width: int,
        encoder_depth: int,
        encoder_num_heads: int,
        sensor_enabled_embedding_dim: int,
        constellation_data_embedding_dim: int,
        decoder_width: int,
        decoder_depth: int,
        decoder_num_heads: int,
        return_logits: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._return_logits = return_logits

        time_embedding = sinusoidal_position_embedding(
            torch.arange(MAX_TIME_STEP),
            time_embedding_dim,
        )
        self._time_embedding = nn.Parameter(time_embedding)

        self._sensor_type_embedding = nn.Embedding(
            len(SensorType),
            sensor_type_embedding_dim,
        )
        self._encoder = Encoder(
            time_embedding_dim=time_embedding_dim,
            sensor_type_embedding_dim=sensor_type_embedding_dim,
            data_embedding_dim=tasks_data_embedding_dim,
            width=encoder_width,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
        )
        self._decoder = Decoder(
            time_embedding_dim=time_embedding_dim,
            sensor_type_embedding_dim=sensor_type_embedding_dim,
            sensor_enabled_embedding_dim=sensor_enabled_embedding_dim,
            data_embedding_dim=constellation_data_embedding_dim,
            width=decoder_width,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            return_logits=return_logits,
        )
        self._time_model = TimeModel()
        self._time_projection = nn.Linear(1, 1)

        self._time_model.requires_grad_(True)
        self._encoder.requires_grad_(False)
        self._decoder.requires_grad_(False)
        self._time_projection.requires_grad_(True)

        
    def forward(
        self,
        time_steps: torch.Tensor | Iterable[int],
        constellation_sensor_type: torch.Tensor,
        constellation_sensor_enabled: torch.Tensor,
        constellation_data: torch.Tensor,
        constellation_mask: torch.Tensor,
        tasks_sensor_type: torch.Tensor,
        tasks_data: torch.Tensor,
        tasks_mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(time_steps, torch.Tensor):
            time_steps = time_steps.flatten().tolist()
        else:
            time_steps = list(time_steps)

        _, time_mask = self._time_model.predict(
            time_steps,
            constellation_data,
            constellation_mask,
            tasks_data,
            tasks_mask,
        )
        time_mask = time_mask.clamp_min(-100)

        time_mask = einops.rearrange(time_mask, 'b ns nt -> b ns nt 1')
        time_mask = self._time_projection(time_mask)
        time_mask = einops.rearrange(time_mask, 'b ns nt 1 -> b ns nt')

        time_embedding = self._time_embedding[time_steps]

        tasks_sensor_type_embedding = self._sensor_type_embedding(
            tasks_sensor_type,
        )
        hidden_states = self._encoder(
            time_embedding,
            tasks_sensor_type_embedding,
            tasks_data,
            tasks_mask,
        )

        constellation_sensor_type_embedding = self._sensor_type_embedding(
            constellation_sensor_type,
        )
        outputs = self._decoder(
            # return self._decoder(
            time_embedding,
            constellation_sensor_type_embedding,
            constellation_sensor_enabled,
            constellation_data,
            constellation_mask,
            hidden_states,
            tasks_mask,
            time_mask,
        )

        if not self._return_logits:
            x = outputs
            return x
        null_logits, logits = outputs

        # if self.with_time_model:
        #     _, pred_mask = self._time_model.predict(
        #         time_steps,
        #         constellation_data,
        #         constellation_mask,
        #         tasks_data,
        #         tasks_mask,
        #     )
        #     pred_mask = pred_mask.sigmoid() < 0.001
        #     # pred_mask = pred_mask.any(-1)
        #     # logits[pred_mask] = float('-inf')
        #     GLOBALS['pred_mask'] = pred_mask

        return null_logits, logits


class DiversityLoss(MSELoss):

    def forward(  # type: ignore[override]
        self,
        logits: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        p = logits.softmax(-1)
        p = p[..., 1:]
        counts = p.sum(-2)
        counts = counts[counts > 1.]
        return super().forward(counts, torch.ones_like(counts))


@ConstellationModelRegistry.register_()
class Model(nn.Module):

    def __init__(
        self,
        *args,
        sensor_type_embedding_dim: int = 128,
        tasks_data_embedding_dim: int = 128,
        encoder_width: int = 512,
        encoder_depth: int = 12,
        encoder_num_heads: int = 16,
        sensor_enabled_embedding_dim: int = 128,
        constellation_data_embedding_dim: int = 128,
        decoder_width: int = 512,
        decoder_depth: int = 12,
        decoder_num_heads: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._transformer = Transformer(
            sensor_type_embedding_dim=sensor_type_embedding_dim,
            tasks_data_embedding_dim=tasks_data_embedding_dim,
            encoder_width=encoder_width,
            encoder_depth=encoder_depth,
            encoder_num_heads=encoder_num_heads,
            sensor_enabled_embedding_dim=sensor_enabled_embedding_dim,
            constellation_data_embedding_dim=constellation_data_embedding_dim,
            decoder_width=decoder_width,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
        )
        self._ce_loss = CrossEntropyLoss()

    def predict(self, *args, **kwargs) -> torch.Tensor:
        null_logits, logits = self._transformer(*args, **kwargs)
        null_logits = einops.rearrange(null_logits, 'b ns -> b ns 1')
        logits = torch.cat((null_logits, logits), -1)
        return logits

    def forward(
        self,
        runner: BaseRunner[nn.Module],
        batch: Batch,
        memo: Memo,
    ) -> Memo:
        log: Memo | None = memo.get('log')
        tensorboard: TensorBoardCallback | None = memo.get('tensorboard')

        batch = Batch(*batch)  # for PrefetchDataLoader
        memo['actions_task_id'] = einops.rearrange(
            batch.actions_task_id + 1,
            'b ns -> (b ns)',
        )

        logits = self.predict(
            batch.time_steps,
            batch.constellation_sensor_type,
            batch.constellation_sensor_enabled,
            batch.constellation_data,
            batch.constellation_mask,
            batch.tasks_sensor_type,
            batch.tasks_data,
            batch.tasks_mask,
        )
        memo['logits'] = einops.rearrange(logits, 'b ns nt -> (b ns) nt')

        ce_loss = self._ce_loss(
            einops.rearrange(logits, 'b ns nt -> (b ns) nt'),
            einops.rearrange(batch.actions_task_id + 1, 'b ns -> (b ns)'),
        )
        loss = ce_loss

        memo['loss'] = loss

        tensors: dict[str, torch.Tensor] = dict(loss=loss, ce_loss=ce_loss)
        if log is not None:
            log.update({k: f'{v:.3f}' for k, v in tensors.items()})
        if tensorboard is not None:
            for k, v in tensors.items():
                tensorboard.summary_writer.add_scalar(
                    tensorboard.tag(k),
                    v.float(),
                    runner.iter_,
                )

        return memo
