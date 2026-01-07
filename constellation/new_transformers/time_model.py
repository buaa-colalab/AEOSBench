import bisect
from collections import UserList
import dataclasses
import random
from typing import Any, Iterable, NamedTuple, TypeVar, cast

import einops
import torch
from todd.models.losses import BCEWithLogitsLoss, MSELoss
from todd.models.modules import sinusoidal_position_embedding
from todd.patches.py_ import get_, json_load
from todd.registries import CollateRegistry
from todd.runners import BaseRunner, Memo
from todd.runners.callbacks import TensorBoardCallback
from todd.runners.memo import Memo
from todd.runners.metrics import Metric
from todd.runners.registries import MetricRegistry
from todd.utils import NestedTensorCollectionUtils
from torch import nn

from constellation import (
    ANNOTATIONS_ROOT,
    CONSTELLATIONS_ROOT,
    MAX_TIME_STEP,
    STATISTICS_PATH,
    TASKSETS_ROOT,
    TRAJECTORIES_ROOT,
)
from constellation.data import Constellation, TaskSet

from .dataset import (
    DynamicConstellationData,
    DynamicTasksetData,
    Statistics,
    TrajectoryData,
)
from .registries import (
    ConstellationDatasetRegistry,
    ConstellationModelRegistry,
)
from .constants import SATELLITE_DIM, TASK_DIM

import pathlib

# TODO: delete
TRAJECTORIES_ROOT = pathlib.Path('data/trajectories.tabu.1')

TIME_SCALE = 50

T = TypeVar('T', bound=nn.Module)


class BaseMetric(Metric[T]):

    def __init__(
        self,
        *args,
        threshold: float,
        logits: str,
        target: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._threshold = threshold
        self._logits = logits
        self._target = target

    def _preprocess(
        self,
        batch: Any,
        memo: Memo,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        logits: torch.Tensor = get_(memo, self._logits)
        target: torch.Tensor = get_(memo, self._target)
        assert logits.dtype.is_floating_point
        assert target.dtype is torch.bool
        pred = logits.sigmoid() >= self._threshold
        tp = (pred & target).sum()
        fp = (pred & ~target).sum()
        fn = (~pred & target).sum()
        return tp, fp, fn, logits.numel()


@MetricRegistry.register_()
class TimeTPRMetric(BaseMetric[T]):

    def _forward(self, batch: Any, memo: Memo) -> tuple[torch.Tensor, Memo]:
        tp, _, _, n = self._preprocess(batch, memo)
        return tp / n, memo


@MetricRegistry.register_()
class TimeFPRMetric(BaseMetric[T]):

    def _forward(self, batch: Any, memo: Memo) -> tuple[torch.Tensor, Memo]:
        _, fp, _, n = self._preprocess(batch, memo)
        return fp / n, memo


@MetricRegistry.register_()
class TimePrecisionMetric(BaseMetric[T]):

    def _forward(self, batch: Any, memo: Memo) -> tuple[torch.Tensor, Memo]:
        tp, fp, _, _ = self._preprocess(batch, memo)
        precision = tp / (tp + fp + 1e-6)
        return precision, memo


@MetricRegistry.register_()
class TimeRecallMetric(BaseMetric[T]):

    def _forward(self, batch: Any, memo: Memo) -> tuple[torch.Tensor, Memo]:
        tp, _, fn, _ = self._preprocess(batch, memo)
        recall = tp / (tp + fn + 1e-6)
        return recall, memo


@dataclasses.dataclass(frozen=True)
class TimeSpan:
    start_time: int
    end_time: int
    satellite_id: int
    task_id: int

    @property
    def length(self) -> int:
        return self.end_time - self.start_time


class TimeSpans(UserList[TimeSpan]):

    def __init__(self) -> None:
        super().__init__()
        self._offsets: list[int] = []

    def append(self, item: TimeSpan) -> None:
        self._offsets.append(self.total_length)
        super().append(item)

    @property
    def total_length(self) -> int:
        return 0 if len(self) == 0 else self[-1].length + self._offsets[-1]

    def _to_data(
        self,
        index: int,
        *,
        with_duration: bool = True,
    ) -> tuple[int, int, int, int]:
        i = bisect.bisect(self._offsets, index) - 1
        time_span = self[i]
        time_step = time_span.start_time + index - self._offsets[i]

        if with_duration:
            duration = time_span.end_time - time_step
            if duration > 2 * TIME_SCALE:
                duration = -TIME_SCALE
        else:
            duration = -TIME_SCALE

        return time_step, duration, time_span.satellite_id, time_span.task_id

    def sample_data(self, n: int, **kwargs) -> torch.Tensor:
        if self.total_length > n:
            indices = random.sample(range(self.total_length), n)
        else:
            indices = list(range(self.total_length))
        return torch.tensor(
            [self._to_data(i, **kwargs) for i in indices],
            dtype=torch.int,
        )


class Batch(NamedTuple):
    time_steps: torch.Tensor
    durations: torch.Tensor
    constellation_data: torch.Tensor
    tasks_data: torch.Tensor


@ConstellationDatasetRegistry.register_()
class TimeDataset(torch.utils.data.Dataset[Batch]):

    def __init__(
        self,
        *args,
        split: str,
        annotation_file: str | None = None,
        batch_size: int,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._split = split

        if annotation_file is None:
            annotation_file = f'{split}.json'
        self._annotations: list[int] = json_load(
            str(ANNOTATIONS_ROOT / annotation_file),
        )

        self._batch_size = batch_size

        if normalize:
            self._statistics: Statistics = torch.load(
                STATISTICS_PATH,
                weights_only=False,
            )

        self._nested_tensor_collection_utils = NestedTensorCollectionUtils()

    @property
    def normalize(self) -> bool:
        return hasattr(self, '_statistics')

    def __len__(self) -> int:
        return len(self._annotations)

    def _load_constellation(
        self,
        constellation: DynamicConstellationData,
        id_: int,
    ) -> torch.Tensor:
        dynamic_data = constellation['data']

        constellation_path = (
            CONSTELLATIONS_ROOT / self._split / f'{id_ // 1000:02}'
            / f'{id_:05}.json'
        )
        _, static_data = Constellation.load(
            str(constellation_path),
        ).static_to_tensor()

        static_data = einops.repeat(
            static_data,
            'ns nd -> t ns nd',
            t=dynamic_data.shape[0],
        )
        data = torch.cat([static_data, dynamic_data], -1)

        return data

    def _load_tasks(
        self,
        taskset: DynamicTasksetData,
        id_: int,
    ) -> torch.Tensor:
        progress = taskset['progress']
        t = progress.shape[0]

        taskset_path = (
            TASKSETS_ROOT / self._split / f'{id_ // 1000:02}'
            / f'{id_:05}.json'
        )
        _, static_data = TaskSet.load(str(taskset_path)).to_tensor()

        static_data = einops.repeat(static_data, 'nt nd -> t nt nd', t=t)

        static_data = static_data.clone()  # for in-place modification
        time_steps = einops.rearrange(torch.arange(t), 't -> t 1')
        static_data[..., 0] -= time_steps
        static_data[..., 1] -= time_steps

        dynamic_data = einops.rearrange(progress, 't nt -> t nt 1')

        data = torch.cat([static_data, dynamic_data], -1)

        return data

    def _append_time_spans(
        self,
        positives: TimeSpans,
        negatives: TimeSpans,
        satellite_id: int,
        actions: torch.Tensor,  # (t, ), int
        action_changed: torch.Tensor,  # (t, ), bool
        consecutive_visible: torch.Tensor,  # (t, ), bool
    ) -> None:
        a = b = 0
        while True:
            b += 1
            if b >= actions.shape[0]:
                break

            if not action_changed[b] and not consecutive_visible[b]:
                continue

            time_span = TimeSpan(
                a,
                b,
                satellite_id,
                cast(int, actions[a].item()),
            )
            if action_changed[b]:
                negatives.append(time_span)
            else:  # consecutive_visible[b]
                positives.append(time_span)
                while b < actions.shape[0] and not action_changed[b]:
                    b += 1
            a = b

    def _parse_time_spans(
        self,
        actions: torch.Tensor,  # (t, ns), int
        is_visible: torch.Tensor,  # (t, ns, nt), bool
    ) -> tuple[TimeSpans, TimeSpans]:
        action_changed = torch.ones_like(actions, dtype=torch.bool)
        action_changed[1:] = (actions[1:] != actions[:-1])

        is_visible = torch.gather(
            is_visible,
            -1,
            einops.rearrange(actions.clamp(0), 't ns -> t ns 1'),
        )
        is_visible = einops.rearrange(is_visible, 't ns 1 -> t ns')
        is_visible[actions == -1] = False

        consecutive_visible = torch.zeros_like(is_visible)
        consecutive_visible[:-2] = (
            is_visible[:-2] & is_visible[1:-1] & is_visible[2:]
            & ~action_changed[:-2] & ~action_changed[1:-1]
            & ~action_changed[2:]
        )

        positives = TimeSpans()
        negatives = TimeSpans()
        for satellite_id in range(actions.shape[1]):
            self._append_time_spans(
                positives,
                negatives,
                satellite_id,
                actions[:, satellite_id],
                action_changed[:, satellite_id],
                consecutive_visible[:, satellite_id],
            )

        return positives, negatives

    def __getitem__(self, index: int) -> Batch:
        id_ = self._annotations[index]

        trajectory: TrajectoryData = torch.load(
            TRAJECTORIES_ROOT / self._split / f'{id_ // 1000:02}'
            / f'{id_:05}.pth',
        )

        constellation_data = self._load_constellation(
            trajectory['constellation'],
            id_,
        )
        tasks_data = self._load_tasks(
            trajectory['taskset'],
            id_,
        )

        if self.normalize:
            constellation_data = (
                (constellation_data - self._statistics.constellation_mean) /
                (self._statistics.constellation_std + 1e-6)
            )
            tasks_data = ((tasks_data - self._statistics.taskset_mean) /
                          (self._statistics.taskset_std + 1e-6))

        positive_time_spans, negative_time_spans = self._parse_time_spans(
            trajectory['actions']['task_id'],
            trajectory['is_visible'],
        )

        positive_data = positive_time_spans.sample_data(self._batch_size // 2)
        negative_data = negative_time_spans.sample_data(
            self._batch_size // 2,
            with_duration=False,
        )
        data = torch.cat([positive_data, negative_data])

        time_steps, durations, satellite_ids, task_ids = data.unbind(-1)

        batch = Batch(
            time_steps,
            durations / TIME_SCALE,
            constellation_data[time_steps, satellite_ids],
            tasks_data[time_steps, task_ids],
        )

        return batch


@CollateRegistry.register_()
def time_collate_fn(batch: list[Batch]) -> Batch:
    return Batch(*map(torch.cat, zip(*batch)))


@ConstellationModelRegistry.register_()
class TimeModel(nn.Module):

    def __init__(
        self,
        *args,
        input_dim: int = SATELLITE_DIM + TASK_DIM,
        time_embedding_dim: int = 64,
        hidden_dim: int = 1024,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        time_embedding = sinusoidal_position_embedding(
            torch.arange(MAX_TIME_STEP),
            time_embedding_dim,
        )
        self._time_embedding = nn.Parameter(time_embedding)
        self._mlp = nn.Sequential(
            nn.Linear(input_dim + time_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self._mse_loss = MSELoss()
        self._bce_loss = BCEWithLogitsLoss()

    def _predict(
        self,
        time_steps: torch.Tensor | list[int],
        constellation_data: torch.Tensor,
        tasks_data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_embedding = self._time_embedding[time_steps]

        data = torch.cat([constellation_data, tasks_data, time_embedding], -1)
        x: torch.Tensor = self._mlp(data)
        pred_time, pred_mask = x.unbind(-1)
        return pred_time, pred_mask

    def predict(
        self,
        time_steps: torch.Tensor | Iterable[int],
        constellation_data: torch.Tensor,
        constellation_mask: torch.Tensor,
        tasks_data: torch.Tensor,
        tasks_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ns = constellation_data.shape[1]
        nt = tasks_data.shape[1]

        mask = (
            einops.rearrange(constellation_mask, 'b ns -> b ns 1')
            & einops.rearrange(tasks_mask, 'b nt -> b 1 nt')
        )

        if not isinstance(time_steps, torch.Tensor):
            time_steps = mask.new_tensor(time_steps, dtype=torch.int)
        time_steps = einops.repeat(
            time_steps,
            'b -> b ns nt',
            ns=ns,
            nt=nt,
        )[mask]
        constellation_data = einops.repeat(
            constellation_data,
            'b ns d -> b ns nt d',
            nt=nt,
        )[mask]
        tasks_data = einops.repeat(
            tasks_data,
            'b nt d -> b ns nt d',
            ns=ns,
        )[mask]

        pred_time, pred_mask = self._predict(
            time_steps,
            constellation_data,
            tasks_data,
        )
        padded_pred_time = pred_time.new_full((mask.shape[0], ns, nt), -1)
        padded_pred_mask = pred_mask.new_full(
            (mask.shape[0], ns, nt),
            float('-inf'),
        )

        padded_pred_time[mask] = pred_time
        padded_pred_mask[mask] = pred_mask

        return padded_pred_time * TIME_SCALE, padded_pred_mask

    def forward(
        self,
        runner: BaseRunner[nn.Module],
        batch: Batch,
        memo: Memo,
    ) -> Memo:
        log: Memo | None = memo.get('log')
        tensorboard: TensorBoardCallback | None = memo.get('tensorboard')

        batch = Batch(*batch)  # for PrefetchDataLoader

        pred_durations, pred_masks = self._predict(
            batch.time_steps,
            batch.constellation_data,
            batch.tasks_data,
        )
        memo.update(pred_mask=pred_masks)

        gt_masks = batch.durations >= 0

        if gt_masks.any():
            mse_loss = self._mse_loss(
                pred_durations[gt_masks],
                batch.durations[gt_masks].float(),
            )
        else:
            mse_loss = 0.

        bce_loss = self._bce_loss(
            pred_masks,
            gt_masks.float(),
            mask=torch.where(
                gt_masks,
                gt_masks.sum() / (~gt_masks).sum(),
                1.,
            ),
        )

        memo.update(pred_masks=pred_masks, gt_masks=gt_masks)

        loss = mse_loss + bce_loss * 2
        memo['loss'] = loss

        tensors: dict[str, torch.Tensor] = dict(
            loss=loss,
            mse_loss=mse_loss,
            bce_loss=bce_loss,
        )
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
