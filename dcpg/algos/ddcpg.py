from collections import deque
import copy

import torch
from torch.distributions import kl_divergence, Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from dcpg.distributions import FixedCategorical


class Buffer:
    def __init__(self, buffer_size, device):
        self.segs = deque(maxlen=buffer_size)
        self.device = device

    def __len__(self):
        return len(self.segs)

    def insert(self, seg):
        self.segs.append(seg)

    def reset(self):
        self.segs = []

    def feed_forward_generator(self, num_mini_batch=None, mini_batch_size=None):
        num_processes = self.segs[0]["obs"].size(1)
        num_segs = len(self.segs)
        batch_size = num_processes * num_segs

        if mini_batch_size is None:
            mini_batch_size = num_processes // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True
        )

        for indices in sampler:
            obs = []
            actions = []
            returns = []

            for idx in indices:
                process_idx = idx // num_segs
                seg_idx = idx % num_segs

                seg = self.segs[seg_idx]
                obs.append(seg["obs"][:, process_idx])
                actions.append(seg["actions"][:, process_idx])
                returns.append(seg["returns"][:, process_idx])

            obs_batch = torch.stack(obs, dim=1).to(self.device)
            actions_batch = torch.stack(actions, dim=1).to(self.device)
            returns_batch = torch.stack(returns, dim=1).to(self.device)

            yield obs_batch, actions_batch, returns_batch


class DDCPG:
    """
    Dynamics-aware Delayed Critic Policy Gradient (DDCPG)
    """

    def __init__(
        self,
        actor_critic,
        # PPO params
        clip_param,
        ppo_epoch,
        num_mini_batch,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        # Aux params
        buffer_size=None,
        aux_epoch=None,
        aux_freq=None,
        aux_num_mini_batch=None,
        dyna_loss_coef=None,
        policy_dist_coef=None,
        value_dist_coef=None,
        # Misc
        device=None,
        **kwargs
    ):
        # Actor-critic
        self.actor_critic = actor_critic

        # PPO params
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Aux params
        self.buffer = Buffer(buffer_size=buffer_size, device=device)

        self.aux_epoch = aux_epoch
        self.aux_freq = aux_freq
        self.aux_num_mini_batch = aux_num_mini_batch

        self.dyna_loss_coef = dyna_loss_coef
        self.policy_dist_coef = policy_dist_coef
        self.value_dist_coef = value_dist_coef

        self.num_policy_updates = 0
        self.prev_value_loss_epoch = 0
        self.prev_dyna_loss_epoch = 0
        self.prev_policy_dist_epoch = 0
        self.prev_value_dist_epoch = 0

        self.device = device

        # Optimizers
        self.policy_optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.aux_optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        # Add obs, actions, returns to buffer
        seg = {key: rollouts[key].cpu() for key in ["obs", "actions", "returns"]}
        self.buffer.insert(seg)

        # Update actor-critic
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        value_dist_epoch = 0

        # Clone actor-critic
        old_actor_critic = copy.deepcopy(self.actor_critic)

        for _ in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                num_mini_batch=self.num_mini_batch
            )

            for sample in data_generator:
                # Sample batch
                (
                    obs_batch,
                    _,
                    actions_batch,
                    old_action_log_probs_batch,
                    _,
                    _,
                    _,
                    _,
                    adv_targs,
                    _,
                ) = sample

                # Feed batch to actor-critic
                actor_outputs, critic_outputs = self.actor_critic(obs_batch)
                dists = actor_outputs["dist"]
                values = critic_outputs["value"]

                # Feed batch to old actor-critic
                with torch.no_grad():
                    old_critic_outputs = old_actor_critic.forward_critic(obs_batch)
                old_values = old_critic_outputs["value"]

                # Compute action loss
                action_log_probs = dists.log_probs(actions_batch)
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                ratio_clipped = torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surr1 = ratio * adv_targs
                surr2 = ratio_clipped * adv_targs
                action_loss = -torch.min(surr1, surr2).mean()
                dist_entropy = dists.entropy().mean()

                # Compute value dist
                value_dist = 0.5 * (values - old_values).pow(2).mean()

                # Update parameters
                self.policy_optimizer.zero_grad()
                loss = (
                    action_loss
                    - dist_entropy * self.entropy_coef
                    + value_dist * self.value_dist_coef
                )
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.policy_optimizer.step()

                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                value_dist_epoch += value_dist.item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        value_dist_epoch /= num_updates

        self.num_policy_updates += 1

        # Aux phase
        if self.num_policy_updates % self.aux_freq == 0:
            value_loss_epoch = 0
            dyna_loss_epoch = 0
            policy_dist_epoch = 0

            # Clone actor-critic
            old_actor_critic = copy.deepcopy(self.actor_critic)

            for _ in range(self.aux_epoch):
                data_generator = self.buffer.feed_forward_generator(
                    num_mini_batch=self.aux_num_mini_batch
                )

                for sample in data_generator:
                    # Sample batch
                    obs_batch, actions_batch, returns_batch = sample
                    ns, np = actions_batch.size()[0:2]
                    obs_batch = obs_batch.view(-1, *obs_batch.size()[2:])
                    actions_batch = actions_batch.view(-1, actions_batch.size(-1))
                    returns_batch = returns_batch[:-1].view(-1, 1)
                    batch_size = actions_batch.size(0)

                    # Feed batch to actor-critic
                    actor_outputs, critic_outputs = self.actor_critic(obs_batch)
                    all_dists = actor_outputs["dist"]
                    all_actor_features = actor_outputs["actor_feature"]
                    all_values = critic_outputs["value"]

                    all_logits = all_dists.logits
                    all_logits = all_logits.view(ns + 1, np, -1)
                    logits = all_logits[:-1].view(-1, all_logits.size(-1))
                    dists = FixedCategorical(logits=logits)

                    feature_dim = all_actor_features.size(-1)
                    all_actor_features = all_actor_features.view(ns + 1, np, -1)
                    actor_features = all_actor_features[:-1].view(-1, feature_dim)
                    next_actor_features = all_actor_features[1:].view(-1, feature_dim)

                    all_values = all_values.view(ns + 1, np, -1)
                    values = all_values[:-1].view(-1, all_values.size(-1))

                    # Feed batch to old actor-critic
                    with torch.no_grad():
                        old_actor_outputs = old_actor_critic.forward_actor(obs_batch)
                    all_old_dists = old_actor_outputs["dist"]

                    all_old_logits = all_old_dists.logits
                    all_old_logits = all_old_logits.view(ns + 1, np, -1)
                    old_logits = all_old_logits[:-1].view(-1, all_old_logits.size(-1))
                    old_dists = FixedCategorical(logits=old_logits)

                    # Compute value loss
                    value_loss = 0.5 * (values - returns_batch).pow(2).mean()

                    # Sample OOD actions
                    num_actions = self.actor_critic.num_actions
                    onehot_actions_batch = F.one_hot(
                        actions_batch.squeeze(), num_classes=num_actions
                    )
                    count = torch.bincount(
                        actions_batch.squeeze(), minlength=num_actions
                    )
                    counts = count.unsqueeze(dim=0).repeat(batch_size, 1)
                    counts = counts * (1 - onehot_actions_batch)
                    masks = (torch.sum(counts, dim=1, keepdims=True) == 0).long()
                    counts += masks * (1 - onehot_actions_batch)
                    probs = counts / torch.sum(counts, dim=1, keepdims=True)
                    rand_action_dists = Categorical(probs=probs)
                    rand_actions_batch = rand_action_dists.sample().unsqueeze(dim=1)

                    # Sample OOD next states
                    rand_indices = torch.arange(0, batch_size).to(self.device)
                    rand_indices += torch.randint_like(rand_indices, 1, batch_size)
                    rand_indices %= batch_size
                    rand_next_actor_features = next_actor_features[rand_indices]

                    # Compute Dynamics loss
                    dyna_logits_in = self.actor_critic.forward_dyna(
                        actor_features, actions_batch, next_actor_features
                    )
                    dyna_logits_a_out = self.actor_critic.forward_dyna(
                        actor_features, rand_actions_batch, next_actor_features
                    )
                    dyna_logits_s_out = self.actor_critic.forward_dyna(
                        actor_features, actions_batch, rand_next_actor_features
                    )

                    dyna_loss_in = F.binary_cross_entropy_with_logits(
                        dyna_logits_in,
                        torch.ones_like(dyna_logits_in),
                    )
                    dyna_loss_a_out = F.binary_cross_entropy_with_logits(
                        dyna_logits_a_out,
                        torch.zeros_like(dyna_logits_a_out),
                    )
                    dyna_loss_s_out = F.binary_cross_entropy_with_logits(
                        dyna_logits_s_out,
                        torch.zeros_like(dyna_logits_s_out),
                    )

                    dyna_loss = dyna_loss_in + 0.5 * dyna_loss_a_out + dyna_loss_s_out

                    # Compute policy dist
                    policy_dist = kl_divergence(old_dists, dists).mean()

                    # Update parameters
                    self.aux_optimizer.zero_grad()
                    loss = (
                        value_loss
                        + dyna_loss * self.dyna_loss_coef
                        + policy_dist * self.policy_dist_coef
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.max_grad_norm
                    )
                    self.aux_optimizer.step()

                    value_loss_epoch += value_loss.item()
                    dyna_loss_epoch += dyna_loss.item()
                    policy_dist_epoch += policy_dist.item()

            num_updates = self.aux_epoch * self.aux_num_mini_batch * len(self.buffer)
            value_loss_epoch /= num_updates
            dyna_loss_epoch /= num_updates
            policy_dist_epoch /= num_updates

            self.prev_value_loss_epoch = value_loss_epoch
            self.prev_dyna_loss_epoch = dyna_loss_epoch
            self.prev_policy_dist_epoch = policy_dist_epoch

        else:
            value_loss_epoch = self.prev_value_loss_epoch
            dyna_loss_epoch = self.prev_dyna_loss_epoch
            policy_dist_epoch = self.prev_policy_dist_epoch

        train_statistics = {
            "action_loss": action_loss_epoch,
            "dist_entropy": dist_entropy_epoch,
            "value_loss": value_loss_epoch,
            "dyna_loss": dyna_loss_epoch,
            "policy_dist": policy_dist_epoch,
            "value_dist": value_dist_epoch,
        }

        return train_statistics
