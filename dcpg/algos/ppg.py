from collections import deque
import copy

import torch
from torch.distributions import kl_divergence
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class Buffer:
    def __init__(self, buffer_size, device):
        self.segs = deque(maxlen=buffer_size)
        self.device = device

    def __len__(self):
        return len(self.segs)

    def insert(self, seg):
        self.segs.append(seg)

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
            returns = []

            for idx in indices:
                process_idx = idx // num_segs
                seg_idx = idx % num_segs

                seg = self.segs[seg_idx]
                obs.append(seg["obs"][:, process_idx])
                returns.append(seg["returns"][:, process_idx])

            obs = torch.stack(obs, dim=1).to(self.device)
            returns = torch.stack(returns, dim=1).to(self.device)

            obs_batch = obs[:-1].view(-1, *obs.size()[2:])
            returns_batch = returns[:-1].view(-1, 1)

            yield obs_batch, returns_batch


class PPG:
    """
    Phasic Policy Gradient (PPG)
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
        policy_dist_coef=None,
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
        self.policy_dist_coef = policy_dist_coef

        self.num_policy_updates = 0
        self.prev_aux_value_loss_epoch = 0
        self.prev_policy_dist_epoch = 0

        self.device = device

        # Optimizers
        self.policy_optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.aux_optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        # Add obs and returns to buffer
        seg = {key: rollouts[key].cpu() for key in ["obs", "returns"]}
        self.buffer.insert(seg)

        # PPO phase
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        value_loss_epoch = 0

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
                    returns_batch,
                    _,
                    adv_targs,
                    _,
                ) = sample

                # Feed batch to actor-critic
                actor_outputs, critic_outputs = self.actor_critic(obs_batch)
                dists = actor_outputs["dist"]
                values = critic_outputs["value"]

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

                # Compute value loss
                value_loss = 0.5 * (values - returns_batch).pow(2).mean()

                # Update parameters
                self.policy_optimizer.zero_grad()
                loss = action_loss - dist_entropy * self.entropy_coef + value_loss
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.policy_optimizer.step()

                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                value_loss_epoch += value_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        value_loss_epoch /= num_updates

        self.num_policy_updates += 1

        # Auxiliary phase
        if self.num_policy_updates % self.aux_freq == 0:
            aux_value_loss_epoch = 0
            policy_dist_epoch = 0

            # Clone actor-critic
            old_actor_critic = copy.deepcopy(self.actor_critic)

            for _ in range(self.aux_epoch):
                data_generator = self.buffer.feed_forward_generator(
                    num_mini_batch=self.aux_num_mini_batch
                )

                for sample in data_generator:
                    # Sample batch
                    obs_batch, returns_batch = sample

                    # Feed batch to actor-critic
                    actor_outputs, critic_outputs = self.actor_critic(obs_batch)
                    dists = actor_outputs["dist"]
                    aux_values = actor_outputs["aux_value"]
                    values = critic_outputs["value"]

                    # Feed batch to old actor-critic
                    with torch.no_grad():
                        old_actor_outputs = old_actor_critic.forward_actor(obs_batch)
                    old_dists = old_actor_outputs["dist"]

                    # Compute aux value loss
                    aux_value_loss = 0.5 * (aux_values - returns_batch).pow(2).mean()

                    # Compute value loss
                    value_loss = 0.5 * (values - returns_batch).pow(2).mean()

                    # Compute policy dist
                    policy_dist = kl_divergence(old_dists, dists).mean()

                    # Update parameters
                    self.aux_optimizer.zero_grad()
                    loss = (
                        aux_value_loss
                        + value_loss
                        + policy_dist * self.policy_dist_coef
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.max_grad_norm
                    )
                    self.aux_optimizer.step()

                    aux_value_loss_epoch += aux_value_loss.item()
                    policy_dist_epoch += policy_dist.item()

            num_updates = self.aux_epoch * self.aux_num_mini_batch * len(self.buffer)
            aux_value_loss_epoch /= num_updates
            policy_dist_epoch /= num_updates

            self.prev_aux_value_loss_epoch = aux_value_loss_epoch
            self.prev_policy_dist_epoch = policy_dist_epoch

        else:
            aux_value_loss_epoch = self.prev_aux_value_loss_epoch
            policy_dist_epoch = self.prev_policy_dist_epoch

        train_statistics = {
            "action_loss": action_loss_epoch,
            "dist_entropy": dist_entropy_epoch,
            "value_loss": value_loss_epoch,
            "aux_value_loss": aux_value_loss_epoch,
            "policy_dist": policy_dist_epoch,
        }

        return train_statistics
