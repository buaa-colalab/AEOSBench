import einops
import torch
from torch import nn

from constellation.new_transformers.model import Transformer


class Model(nn.Module):

    def __init__(
        self,
        *args,
        sensor_type_embedding_dim: int = 128,
        tasks_data_embedding_dim: int = 128,
        encoder_width: int = 64,
        encoder_depth: int = 1,
        encoder_num_heads: int = 4,
        sensor_enabled_embedding_dim: int = 128,
        constellation_data_embedding_dim: int = 128,
        decoder_width: int,
        decoder_depth: int = 1,
        decoder_num_heads: int = 4,
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
            return_logits=False,
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        x = self._transformer(*args, **kwargs)
        # x = self._out_projector(x)
        x = einops.reduce(x, 'b nt d -> b d', 'mean')
        return x
