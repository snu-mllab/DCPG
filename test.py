import numpy as np
import torch

from dcpg.envs import make_envs


def evaluate(config, actor_critic, device, test_envs=True, norm_infos=None):
    # Set actor-critic to evaluation mode
    actor_critic.eval()

    # Create environments
    if test_envs:
        envs = make_envs(
            num_envs=config["num_eval_episodes"],
            env_name=config["env_name"],
            num_levels=0,
            start_level=0,
            distribution_mode=config["distribution_mode"],
            normalize_reward=False,
            device=device,
        )
    else:
        envs = make_envs(
            num_envs=config["num_eval_episodes"],
            env_name=config["env_name"],
            num_levels=config["num_levels"],
            start_level=config["start_level"],
            distribution_mode=config["distribution_mode"],
            normalize_reward=False,
            device=device,
        )

    # Initialize environments
    obs = envs.reset()

    # Sample episodes
    episode_rewards = []
    episode_steps = []
    finished = torch.zeros(obs.size(0))

    step = 0

    # Initialize value and return variables
    if norm_infos is not None:
        var, eps, cliprew = norm_infos.values()
        var = torch.tensor(var)
        gamma = 0.999
        disc = 1

        # Initial value estimates
        with torch.no_grad():
            init_value_ests = actor_critic.forward_critic(obs)["value"]
        init_value_ests = torch.squeeze(init_value_ests)

        # Returns
        ret_un_nodisc = torch.zeros(config["num_eval_episodes"])
        ret_n_nodisc = torch.zeros(config["num_eval_episodes"])
        ret_un_disc = torch.zeros(config["num_eval_episodes"])
        ret_n_disc = torch.zeros(config["num_eval_episodes"])

    while not torch.all(finished):
        # Sample action
        with torch.no_grad():
            action, *_ = actor_critic.act(obs)

        # Interact with environments
        obs, reward, *_, infos = envs.step(action)
        step += 1

        # Update returns
        if norm_infos is not None:
            # Calculate Normalized rewards
            reward_n = torch.clip(reward / torch.sqrt(var + eps), -cliprew, cliprew)

            # Update coefficient for discounting
            disc *= gamma

            # Add rewards to returns
            ret_un_nodisc += torch.squeeze(reward) * (1 - finished)
            ret_un_disc += disc * torch.squeeze(reward) * (1 - finished)
            ret_n_nodisc += torch.squeeze(reward_n) * (1 - finished)
            ret_n_disc += disc * torch.squeeze(reward_n) * (1 - finished)

        # Track episode info
        for i, info in enumerate(infos):
            if "episode" in info and not finished[i]:
                episode_rewards.append(info["episode"]["r"])
                episode_steps.append(step)
                finished[i] = 1

    # Close environment
    envs.close()

    # Statistics
    eval_statistics = {
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
    }

    value_statistics = {}
    if norm_infos is not None:
        value_statistics = {
            "init_value_ests": init_value_ests.mean().item(),
            "ret_un_nodisc": ret_un_nodisc.mean().item(),
            "ret_un_disc": ret_un_disc.mean().item(),
            "ret_n_nodisc": ret_n_nodisc.mean().item(),
            "ret_n_disc": ret_n_disc.mean().item(),
        }

    return eval_statistics, value_statistics
