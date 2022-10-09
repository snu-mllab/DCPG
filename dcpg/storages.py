import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.levels = torch.LongTensor(num_steps + 1, num_processes).fill_(0)

        self.num_steps = num_steps
        self.step = 0

    def __getitem__(self, key):
        return getattr(self, key)

    def to(self, device):
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.masks = self.masks.to(device)
        self.levels = self.levels.to(device)

    def insert(
        self, obs, actions, action_log_probs, rewards, value_preds, masks, levels
    ):
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.rewards[self.step].copy_(rewards)
        self.value_preds[self.step].copy_(value_preds)
        self.masks[self.step + 1].copy_(masks)
        self.levels[self.step + 1].copy_(levels)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.levels[0].copy_(self.levels[-1])

    def compute_returns(self, next_value, gamma, gae_lambda):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = (
                self.rewards[step]
                + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                - self.value_preds[step]
            )
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]

    def compute_advantages(self):
        self.advantages = self.returns[:-1] - self.value_preds[:-1]
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-5
        )

    def feed_forward_generator(self, num_mini_batch=None, mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(
                    num_processes, num_steps, num_processes * num_steps, num_mini_batch
                )
            )
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True,
        )

        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            next_obs_batch = self.obs[1:].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            rewards_batch = self.rewards.view(-1, 1)[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            returns_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[1:].view(-1, 1)[indices]
            levels_batch = self.levels[:-1].view(-1, 1)[indices]

            if hasattr(self, "advantages"):
                adv_targs = self.advantages.view(-1, 1)[indices]
            else:
                adv_targs = None

            yield (
                obs_batch,
                next_obs_batch,
                actions_batch,
                old_action_log_probs_batch,
                rewards_batch,
                value_preds_batch,
                returns_batch,
                masks_batch,
                adv_targs,
                levels_batch,
            )
