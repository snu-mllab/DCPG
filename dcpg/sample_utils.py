import torch


def sample_episodes(envs, rollouts, actor_critic):
    episode_rewards = []

    # Sample episodes
    for step in range(rollouts.num_steps):
        # Sample action
        with torch.no_grad():
            action, action_log_prob, value = actor_critic.act(rollouts.obs[step])

        # Interact with environment
        obs, reward, done, infos = envs.step(action)
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        levels = torch.LongTensor([info["level_seed"] for info in infos])

        # Insert obs, action and reward into rollout storage
        rollouts.insert(obs, action, action_log_prob, reward, value, masks, levels)

        # Track episode info
        for info in infos:
            if "episode" in info.keys():
                episode_rewards.append(info["episode"]["r"])

    return episode_rewards
