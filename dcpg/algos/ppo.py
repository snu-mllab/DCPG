
import torch
import torch.nn as nn
import torch.optim as optim


class PPO:
    """
    Proximal Policy Optimization (PPO)
    """

    def __init__(
        self,
        actor_critic,
        # PPO params
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        **kwargs
    ):
        # Actor-critic
        self.actor_critic = actor_critic

        # PPO params
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Optimizers
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        # Update actor-critic
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        value_loss_epoch = 0
        
        for _ in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                num_mini_batch=self.num_mini_batch,
            )

            for sample in data_generator:
                # Sample batch
                (
                    obs_batch,
                    _,
                    actions_batch,
                    old_action_log_probs_batch,
                    _,
                    value_preds_batch,
                    returns_batch,
                    _,
                    adv_targs,
                    _
                ) = sample

                # Feed batch to actor-critic
                actor_outputs, values_outputs = self.actor_critic(obs_batch)
                dists = actor_outputs["dist"]
                values = values_outputs["value"]

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
                value_pred_clipped = value_preds_batch + torch.clamp(
                    values - value_preds_batch, -self.clip_param, self.clip_param
                )
                value_losses = (values - returns_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                
                # Update parameters
                self.optimizer.zero_grad()
                loss = (
                    action_loss
                    - dist_entropy * self.entropy_coef
                    + value_loss * self.value_loss_coef
                )
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                value_loss_epoch += value_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        value_loss_epoch /= num_updates

        train_statistics = {
            "action_loss": action_loss_epoch,
            "dist_entropy": dist_entropy_epoch,
            "value_loss": value_loss_epoch,
        }

        return train_statistics
