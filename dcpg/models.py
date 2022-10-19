from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from dcpg.distributions import Categorical, FixedCategorical
from dcpg.utils import init

init_ = lambda m: init(
    m,
    nn.init.orthogonal_,
    lambda x: nn.init.constant_(x, 0),
)

init_relu_ = lambda m: init(
    m,
    nn.init.orthogonal_,
    lambda x: nn.init.constant_(x, 0),
    nn.init.calculate_gain("relu"),
)


def apply_init_(modules: Sequence[nn.Module]):
    """
    Initialize modules
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    """
    Flatten a tensor
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(x.size(0), -1)


class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input: Tensor, dim: int) -> Tuple[int, int]:
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        needed_input = (out_size - 1) * self.stride[dim] + effective_filter_size
        total_padding = max(0, needed_input - input_size)
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input: Tensor) -> Tensor:
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class BasicBlock(nn.Module):
    """
    Residual Network Block
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv1 = Conv2d_tf(
            n_channels,
            n_channels,
            kernel_size=3,
            stride=1,
            padding=(1, 1),
        )
        self.relu = nn.ReLU()
        self.conv2 = Conv2d_tf(
            n_channels,
            n_channels,
            kernel_size=3,
            stride=1,
            padding=(1, 1),
        )

        apply_init_(self.modules())

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity

        return out


class ResNetEncoder(nn.Module):
    """
    Residual Network Encoder
    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        feature_dim: int = 256,
        channels: Sequence[int] = [16, 32, 32],
        widen_factor: int = 1,
    ):
        super().__init__()

        # Layer
        self.feature_dim = feature_dim

        self.layers = nn.ModuleList()
        in_channel = obs_shape[0]
        for channel in channels:
            out_channel = widen_factor * channel
            self.layers.append(self._make_layer(in_channel, out_channel))
            in_channel = out_channel

        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.fc = init_relu_(nn.Linear(widen_factor * 2048, feature_dim))

        apply_init_(self.modules())

    def _make_layer(self, in_channels: int, out_channels: int) -> nn.Sequential:
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        x = self.flatten(x)
        x = self.relu(x)
        x = self.fc(x)
        x = self.relu(x)

        return x


class PPOModel(nn.Module):
    """
    PPO Actor-Critic Model
    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        num_actions: int,
        shared: bool = True,
        **kwargs,
    ):
        super().__init__()

        # Encoder
        self.actor_key = "actor"
        self.critic_key = "actor" if shared else "critic"
        self.true_keys = list(set([self.actor_key, self.critic_key]))

        self.encoders = nn.ModuleDict()
        for key in self.true_keys:
            self.encoders[key] = ResNetEncoder(obs_shape, **kwargs)

        # Actor
        self.actor_feature_dim = self.encoders[self.actor_key].feature_dim
        self.actor_heads = nn.ModuleDict()
        self.actor_heads["dist"] = Categorical(self.actor_feature_dim, num_actions)

        # Critic
        self.critic_feature_dim = self.encoders[self.critic_key].feature_dim
        self.critic_heads = nn.ModuleDict()
        self.critic_heads["value"] = init_(nn.Linear(self.critic_feature_dim, 1))

    def act(
        self,
        obs: Tensor,
        deterministic: bool = False,
    ) -> Union[Tensor, Tensor, Tensor]:
        """
        Sample actions
        """
        # Forward obs
        actor_outputs, critic_outputs = self.forward(obs)

        # Sample actions
        dists = actor_outputs["dist"]
        actions = dists.mode() if deterministic else dists.sample()
        action_log_probs = dists.log_probs(actions)

        # Get values
        values = critic_outputs["value"]

        return actions, action_log_probs, values

    def forward(
        self,
        obs: Tensor,
    ) -> Tuple[Dict[str, Union[Tensor, FixedCategorical]], Dict[str, Tensor]]:
        """
        Forward
        """
        # Encoder
        features = dict()
        for key in self.true_keys:
            features[key] = self.encoders[key](obs)
        actor_features = features[self.actor_key]
        critic_features = features[self.critic_key]

        # Actor
        actor_outputs = dict()
        actor_outputs["actor_feature"] = actor_features
        actor_outputs.update(self.forward_actor_heads(actor_features))

        # Critic
        critic_outputs = dict()
        critic_outputs["critic_feature"] = critic_features
        critic_outputs.update(self.forward_critic_heads(critic_features))

        return actor_outputs, critic_outputs

    def forward_actor(self, obs: Tensor) -> Dict[str, Union[Tensor, FixedCategorical]]:
        """
        Forward actor
        """
        # Encoder
        actor_features = self.encoders[self.actor_key](obs)

        # Actor
        actor_outputs = dict()
        actor_outputs["actor_feature"] = actor_features
        actor_outputs.update(self.forward_actor_heads(actor_features))

        return actor_outputs

    def forward_critic(self, obs: Tensor) -> Dict[str, Tensor]:
        """
        Forward critic
        """
        # Encoder
        critic_features = self.encoders[self.critic_key](obs)

        # Critic
        critic_outputs = dict()
        critic_outputs["critic_feature"] = critic_features
        critic_outputs.update(self.forward_critic_heads(critic_features))

        return critic_outputs

    def forward_actor_heads(
        self,
        actor_features: Tensor,
    ) -> Dict[str, Union[Tensor, FixedCategorical]]:
        """
        Forward policy head
        """
        actor_head_outputs = dict()
        actor_head_outputs["dist"] = self.actor_heads["dist"](actor_features)

        return actor_head_outputs

    def forward_critic_heads(self, critic_features: Tensor) -> Dict[str, Tensor]:
        """
        Forward value head
        """
        critic_head_outputs = dict()
        critic_head_outputs["value"] = self.critic_heads["value"](critic_features)

        return critic_head_outputs


class PPGModel(PPOModel):
    """
    PPG Actor-Critic Model
    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        num_actions: int,
        shared: bool = False,
        **kwargs,
    ):
        super().__init__(obs_shape, num_actions, shared, **kwargs)

        # Actor
        self.actor_heads["aux_value"] = init_(nn.Linear(self.actor_feature_dim, 1))

    def forward_actor_heads(
        self,
        actor_features: Tensor,
    ) -> Dict[str, Union[Tensor, FixedCategorical]]:
        """
        Forward policy head + aux value head
        """
        actor_head_outputs = super().forward_actor_heads(actor_features)
        actor_head_outputs["aux_value"] = self.actor_heads["aux_value"](actor_features)

        return actor_head_outputs


class DAACModel(PPOModel):
    """
    DAAC Actor-Critic Model
    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        num_actions: int,
        shared: bool = False,
        **kwargs,
    ):
        super().__init__(obs_shape, num_actions, shared, **kwargs)
        self.num_actions = num_actions

        # Actor
        self.actor_heads["adv"] = init_(
            nn.Linear(self.actor_feature_dim + num_actions, 1)
        )

    def forward(
        self, 
        obs: Tensor, 
        actions: Optional[Tensor] = None,
    ) -> Tuple[Dict[str, Union[Tensor, FixedCategorical]], Dict[str, Tensor]]:
        """
        Forward
        """
        # Encoder
        features = dict()
        for key in self.true_keys:
            features[key] = self.encoders[key](obs)
        actor_features = features[self.actor_key]
        critic_features = features[self.critic_key]

        # Actor
        actor_outputs = dict()
        actor_outputs["actor_feature"] = actor_features
        actor_outputs.update(self.forward_actor_heads(actor_features, actions))

        # Critic
        critic_outputs = dict()
        critic_outputs["critic_feature"] = critic_features
        critic_outputs.update(self.forward_critic_heads(critic_features))

        return actor_outputs, critic_outputs

    def concat_sa(self, features: Tensor, actions: Tensor) -> Tensor:
        """
        Concatenate (s, a)
        """
        onehot_actions = F.one_hot(actions.squeeze(dim=-1), self.num_actions).float()
        return torch.cat([features, onehot_actions], dim=-1)

    def forward_actor(
        self,
        obs: Tensor,
        actions: Optional[Tensor] = None,
    ) -> Dict[str, Union[Tensor, FixedCategorical]]:
        """
        Forward actor
        """
        # Encoder
        actor_features = self.encoders[self.actor_key](obs)

        # Actor
        actor_outputs = dict()
        actor_outputs["actor_feature"] = actor_features
        actor_outputs.update(self.forward_actor_heads(actor_features, actions))

        return actor_outputs

    def forward_actor_heads(
        self,
        actor_features: Tensor,
        actions: Tensor,
    ) -> Dict[str, Union[Tensor, FixedCategorical]]:
        """
        Forward policy head + adv head
        """
        actor_head_outputs = super().forward_actor_heads(actor_features)
        if actions is None:
            actions = actor_head_outputs["dist"].sample()
        inputs_cat = self.concat_sa(actor_features, actions)
        actor_head_outputs["adv"] = self.actor_heads["adv"](inputs_cat)

        return actor_head_outputs


class PPODynaModel(PPOModel):
    """
    PPO + Dynamics Model
    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        num_actions: int,
        shared: bool = True,
        **kwargs,
    ):
        super().__init__(obs_shape, num_actions, shared, **kwargs)
        self.num_actions = num_actions

        # Dynamics
        self.dyna_layer = nn.Sequential(
            init_relu_(nn.Linear(2 * self.actor_feature_dim + num_actions, 256)),
            nn.ReLU(),
            init_relu_(nn.Linear(256, 256)),
            nn.ReLU(),
            init_(nn.Linear(256, 1)),
        )

    def concat_sas(
        self,
        features: Tensor,
        actions: Tensor,
        next_features: Tensor,
    ) -> Tensor:
        """
        Concatenate (s, a, s')
        """
        onehot_actions = F.one_hot(actions.squeeze(dim=-1), self.num_actions).float()
        return torch.cat([features, onehot_actions, next_features], dim=-1)

    def forward_dyna(
        self,
        features: Tensor,
        actions: Tensor,
        next_features: Tensor,
    ) -> Tensor:
        """
        Forward dynamics head
        """
        inputs_cat = self.concat_sas(features, actions, next_features)
        dyna_logits = self.dyna_layer(inputs_cat)
        return dyna_logits
