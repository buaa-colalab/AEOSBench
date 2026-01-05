from typing import NamedTuple, TypedDict, cast

import einops
import todd
import torch
from constellation.new_transformers import Model as ActorModel
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from todd.patches.torch import load_state_dict, load_state_dict_
from torch import nn
from torch.distributions import Categorical

from .critic import Model as CriticModel
from .environment import MAX_NUM_SATELLITES, MAX_NUM_TASKS

VALUE_WIDTH = 64


class Observation(TypedDict):
    num_satellites: torch.Tensor  # b x MAX_NUM_SATELLITES, one-hot
    num_tasks: torch.Tensor  # b x MAX_NUM_TASKS, one-hot
    time_step: torch.Tensor  # b x MAX_TIME_STEP, one-hot
    constellation_sensor_type: torch.Tensor  # b x (MAX_NUM_SATELLITES x len(SensorType)), one-hot # noqa: E501
    constellation_sensor_enabled: torch.Tensor  # b x MAX_NUM_SATELLITES
    constellation_data: torch.Tensor  # b x MAX_NUM_SATELLITES x SATELLITE_DIM
    tasks_sensor_type: torch.Tensor  # b x (MAX_NUM_TASKS x len(SensorType)), one-hot # noqa: E501
    tasks_data: torch.Tensor  # b x MAX_NUM_TASKS x TASK_DIM


class Batch(NamedTuple):
    time_step: torch.Tensor
    constellation_sensor_type: torch.Tensor
    constellation_sensor_enabled: torch.Tensor
    constellation_data: torch.Tensor
    constellation_mask: torch.Tensor
    tasks_sensor_type: torch.Tensor
    tasks_data: torch.Tensor
    tasks_mask: torch.Tensor


class FeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            features_dim=VALUE_WIDTH,  # type: ignore[misc]
            **kwargs,
        )

    def forward(self, observation: Observation) -> Batch:
        num_satellites = observation['num_satellites'].argmax(-1)
        num_tasks = observation['num_tasks'].argmax(-1)
        time_step = observation['time_step'].argmax(-1)

        max_num_satellites = cast(int, num_satellites.max().item())
        max_num_tasks = cast(int, num_tasks.max().item())

        constellation_sensor_type = einops.rearrange(
            observation['constellation_sensor_type'],
            'b (ns st) -> b ns st',
            ns=MAX_NUM_SATELLITES,
        ).argmax(-1)
        constellation_sensor_type = (
            constellation_sensor_type[:, :max_num_satellites].int()
        )

        constellation_sensor_enabled = (
            observation['constellation_sensor_enabled']
            [:, :max_num_satellites].int()
        )

        constellation_data = (
            observation['constellation_data'][:, :max_num_satellites]
        )

        constellation_mask = torch.zeros(
            [num_satellites.shape[0], max_num_satellites],
            dtype=torch.bool,
        )
        for i, n in enumerate(num_satellites):
            constellation_mask[i, :n] = True

        tasks_sensor_type = einops.rearrange(
            observation['tasks_sensor_type'],
            '... (nt st) -> ... nt st',
            nt=MAX_NUM_TASKS,
        ).argmax(-1)
        tasks_sensor_type = tasks_sensor_type[:, :max_num_tasks].int()

        tasks_data = observation['tasks_data'][:, :max_num_tasks]

        tasks_mask = torch.zeros(
            [num_tasks.shape[0], max_num_tasks],
            dtype=torch.bool,
        )
        for i, n in enumerate(num_tasks):
            tasks_mask[i, :n] = True

        return Batch(
            time_step=time_step,
            constellation_sensor_type=constellation_sensor_type,
            constellation_sensor_enabled=constellation_sensor_enabled,
            constellation_data=constellation_data,
            constellation_mask=constellation_mask,
            tasks_sensor_type=tasks_sensor_type,
            tasks_data=tasks_data,
            tasks_mask=tasks_mask,
        )


class ActorCritic(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latent_dim_pi = MAX_NUM_TASKS
        self.latent_dim_vf = VALUE_WIDTH

        self.actor = ActorModel()
        self.critic = CriticModel(decoder_width=VALUE_WIDTH)

    def forward(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(batch), self.forward_critic(batch)

    def forward_actor(self, batch: Batch) -> torch.Tensor:
        if todd.Store.cuda:
            batch = Batch(*[tensor.cuda() for tensor in batch])
        logits = self.actor.predict(*batch)

        padding = logits.new_full(
            (logits.shape[0], MAX_NUM_SATELLITES, MAX_NUM_TASKS),
            float('-inf'),
        )
        padding[..., 0] = 0
        padding[:, :logits.shape[1], :logits.shape[2]] = logits

        return padding

    def forward_critic(self, batch: Batch) -> torch.Tensor:
        if todd.Store.cuda:
            batch = Batch(*[tensor.cuda() for tensor in batch])
        return self.critic(*batch)


class Policy(ActorCriticPolicy):

    def __init__(self, *args, load_model_from: list[str], **kwargs) -> None:
        super().__init__(  # type: ignore[misc]
            *args,
            ortho_init=False,
            features_extractor_class=FeatureExtractor,
            **kwargs,
        )
        if load_model_from:
            load_state_dict(
                self.mlp_extractor.actor,  # type: ignore[arg-type]
                load_state_dict_(load_model_from),  # type: ignore[arg-type]
                strict=False,
            )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ActorCritic()  # type: ignore[assignment]

    def _get_action_dist_from_latent(
        self,
        latent_pi: torch.Tensor,
    ) -> Distribution:
        self.action_dist.distribution = [
            Categorical(logits=logits) for logits in latent_pi.unbind(1)
        ]
        return self.action_dist
