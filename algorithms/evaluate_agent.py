import gymnasium as gym
import numpy as np
import torch

from algorithms.utils import transform_obs


def evaluate(
    env_type,
    model_path,
    make_env,
    env_id,
    eval_episodes,
    run_name,
    Model,
    device,
    gamma=0.99,        
    goal_size=None,      
    capture_video=False,
    deterministic=True,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_type, env_id, 0, capture_video, f"{run_name}-eval")],
        autoreset_mode=gym.vector.vector_env.AutoresetMode.SAME_STEP,
    )

    state_dict = torch.load(model_path, map_location=device)
    if goal_size is None:
        obs_dim = int(np.array(envs.single_observation_space["observation"].shape).prod())
        in_features = int(state_dict["actor_mean.0.weight"].shape[1])
        goal_size = max(0, in_features - obs_dim)

    agent = Model(envs, goal_size=goal_size).to(device)
    agent.load_state_dict(state_dict)
    agent.eval()

    action_low = envs.single_action_space.low
    action_high = envs.single_action_space.high

    episodic_returns = []
    successes = []
    obs, _ = envs.reset()
    obs = torch.Tensor(transform_obs(obs, goal_size)).to(device)
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            if deterministic:
                action = agent.actor_mean(obs)
            else:
                action, _, _, _ = agent.get_action_and_value(obs)
        clipped = np.clip(action.cpu().numpy(), action_low, action_high)
        next_obs, _, _, _, infos = envs.step(clipped)

        final_info = infos.get("final_info")
        if isinstance(final_info, dict) and "episode" in final_info:
            episode = final_info["episode"]
            returns = np.asarray(episode["r"]).reshape(-1)
            valid = np.asarray(
                final_info.get("_episode", np.ones(returns.shape, dtype=bool))
            ).reshape(-1)
            success = final_info.get("success")
            success = None if success is None else np.asarray(success).reshape(-1)
            for i, ok in enumerate(valid):
                if not ok:
                    continue
                ret = float(returns[i])
                episodic_returns.append(ret)
                line = f"eval_episode={len(episodic_returns)}, episodic_return={ret:.3f}"
                if success is not None:
                    is_success = bool(success[i])
                    successes.append(is_success)
                    line += f", success={is_success}"
                print(line)
        obs = torch.Tensor(transform_obs(next_obs, goal_size)).to(device)

    envs.close()

    mean_return = float(np.mean(episodic_returns)) if episodic_returns else float("nan")
    summary = f"[eval] episodes={len(episodic_returns)} mean_return={mean_return:.3f}"
    if successes:
        summary += f" success_rate={float(np.mean(successes)):.1%}"
    print(summary)
    return episodic_returns
