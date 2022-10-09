import torch
import torch.nn as nn
import torch.optim as optim


class DAAC:
    """
    Decoupled Advantage Actor-Critic (DAAC)
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
        value_epoch=None,
        value_freq=None,
        adv_loss_coef=None,
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

        # DAAC params
        self.value_epoch = value_epoch
        self.value_freq = value_freq
        self.adv_loss_coef = adv_loss_coef

        self.num_policy_updates = 0

        # Optimizers
        self.policy_optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.value_optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        # Policy phase
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        adv_loss_epoch = 0

        for _ in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(self.num_mini_batch)

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

                # Feed batch to actor
                actor_outputs = self.actor_critic.forward_actor(
                    obs_batch, actions=actions_batch
                )
                dists = actor_outputs["dist"]
                advs = actor_outputs["adv"]

                # Compute policy loss
                action_log_probs = dists.log_probs(actions_batch)
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                ratio_clipped = torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surr1 = ratio * adv_targs
                surr2 = ratio_clipped * adv_targs
                action_loss = -torch.min(surr1, surr2).mean()
                dist_entropy = dists.entropy().mean()

                # Compute adv loss
                adv_loss = (advs - adv_targs).pow(2).mean()

                # Update parameters
                self.policy_optimizer.zero_grad()
                loss = (
                    action_loss
                    + adv_loss * self.adv_loss_coef
                    - dist_entropy * self.entropy_coef
                )
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.policy_optimizer.step()

                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                adv_loss_epoch += adv_loss.item()

        num_policy_updates = self.ppo_epoch * self.num_mini_batch
        action_loss_epoch /= num_policy_updates
        dist_entropy_epoch /= num_policy_updates
        adv_loss_epoch /= num_policy_updates

        # Value phase
        if self.num_policy_updates % self.value_freq == 0:
            value_loss_epoch = 0

            for _ in range(self.value_epoch):
                data_generator = rollouts.feed_forward_generator(
                    num_mini_batch=self.num_mini_batch
                )

                for sample in data_generator:
                    # Sample batch
                    (
                        obs_batch,
                        _,
                        actions_batch,
                        _,
                        _,
                        value_preds_batch,
                        returns_batch,
                        _,
                        _,
                        _,
                    ) = sample

                    # Feed batch to critic
                    critic_outputs = self.actor_critic.forward_critic(obs_batch)
                    values = critic_outputs["value"]

                    # Compute value loss
                    value_pred_clipped = value_preds_batch + torch.clamp(
                        values - value_preds_batch, -self.clip_param, self.clip_param
                    )
                    value_losses = (values - returns_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                    # Update parameters
                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.max_grad_norm
                    )
                    self.value_optimizer.step()

                    value_loss_epoch += value_loss.item()

            num_value_updates = self.value_epoch * self.num_mini_batch
            value_loss_epoch /= num_value_updates
            self.prev_value_loss_epoch = value_loss_epoch

        else:
            value_loss_epoch = self.prev_value_loss_epoch

        self.num_policy_updates += 1

        train_statistics = {
            "action_loss": action_loss_epoch,
            "dist_entropy": dist_entropy_epoch,
            "value_loss": value_loss_epoch,
            "adv_loss": adv_loss_epoch,
        }

        return train_statistics
