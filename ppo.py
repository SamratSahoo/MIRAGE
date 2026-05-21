import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch._dynamo

from sys import platform
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from algorithms.evaluate_agent import evaluate
from algorithms.utils import make_env
from algorithms.warp_antmaze import WarpAntMazeEnv

os.environ["MUJOCO_GL"] = "glfw" if platform == "darwin" else "osmesa"

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, goal_size=0):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space['observation'].shape).prod() + goal_size, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space['observation'].shape).prod() + goal_size, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class PPOTrainer:
    def __init__(
    self, 
    env_type,
    env_id="spin_rl", 
    num_envs=4, 
    seed=1, 
    num_steps=128, 
    num_minibatches=4,
    exp_name=os.path.basename(__file__)[: -len(".py")],
    learning_rate=2.5e-4,
    anneal_lr=True,
    target_kl=0.1,
    gamma = 0.99,
    gae_lambda = 0.95,
    update_epochs=4,
    norm_adv=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef= 0.01,
    vf_coef= 0.5,
    max_grad_norm=0.5,
    goal_size=0,
    load_default_checkpoint=True,
    track=False,
    wandb_project_name="mirage-ppo",
    wandb_entity=None
    ):
        self.seed = seed
        self.env_id = env_id
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.exp_name = exp_name
        self.batch_size = int(self.num_envs * self.num_steps)
        self.num_minibatches = num_minibatches
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.goal_size = goal_size
        self.track = track
        self.load_default_checkpoint = load_default_checkpoint

        self.run_name = self.exp_name
        self.run_dir = os.path.join("runs", self.run_name)
        self.model_path = os.path.join(self.run_dir, f"{self.exp_name}.cleanrl_model")
        self.best_model_path = os.path.join(self.run_dir, f"{self.exp_name}_best.cleanrl_model")
        self._best_eval_return = None

        self.hyperparams = {
            "env_type": env_type,
            "env_id": env_id,
            "exp_name": exp_name,
            "load_default_checkpoint": load_default_checkpoint,
            "seed": seed,
            "num_envs": num_envs,
            "num_steps": num_steps,
            "batch_size": self.batch_size,
            "num_minibatches": num_minibatches,
            "minibatch_size": self.minibatch_size,
            "update_epochs": update_epochs,
            "learning_rate": learning_rate,
            "anneal_lr": anneal_lr,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "norm_adv": norm_adv,
            "clip_coef": clip_coef,
            "clip_vloss": clip_vloss,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "target_kl": target_kl,
            "goal_size": goal_size,
        }

        if self.track:
            import wandb

            wandb.init(
                project=wandb_project_name,
                entity=wandb_entity,
                sync_tensorboard=True,
                name=self.run_name,
                config=self.hyperparams,
                save_code=True,
            )

        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n"
            + "\n".join(f"|{k}|{v}|" for k, v in self.hyperparams.items()),
        )
        self.env_type = env_type
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.envs = WarpAntMazeEnv(
            env_id=self.env_id,
            num_envs=self.num_envs,
            device=self.device,
            seed=self.seed,
        )

        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.target_kl = target_kl
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef =ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm


    def _evaluate_agent(self, agent, global_step, eval_episodes=10):
        torch.save(agent.state_dict(), self.model_path)
        eval_returns = evaluate(
            self.env_type,
            self.model_path,
            make_env,
            self.env_id,
            eval_episodes=eval_episodes,
            run_name=self.run_name,
            Model=Agent,
            device=self.device,
            gamma=self.gamma,
            goal_size=self.goal_size,
        )
        if not eval_returns:
            return
        eval_returns = np.array(eval_returns, dtype=np.float32)
        mean_return = float(eval_returns.mean())
        self.writer.add_scalar("eval/episodic_return_mean", mean_return, global_step)
        self.writer.add_scalar("eval/episodic_return_std", float(eval_returns.std()), global_step)
        self.writer.add_scalar("eval/episodic_return_min", float(eval_returns.min()), global_step)
        self.writer.add_scalar("eval/episodic_return_max", float(eval_returns.max()), global_step)
        self.writer.add_histogram("eval/episodic_return_hist", eval_returns, global_step)

        if self._best_eval_return is None or mean_return > self._best_eval_return:
            self._best_eval_return = mean_return
            torch.save(agent.state_dict(), self.best_model_path)
            print(
                f"[eval] global_step={global_step} mean_return={mean_return:.3f}  "
                f"->  new best, saved {self.best_model_path}"
            )
        else:
            print(
                f"[eval] global_step={global_step} mean_return={mean_return:.3f}  "
                f"(best={self._best_eval_return:.3f})"
            )
        self.writer.add_scalar("eval/best_episodic_return_mean", self._best_eval_return, global_step)

    def train(self, total_timesteps=10000000, save_model=True, save_freq=0, eval_freq=250000):
        self.num_iterations = total_timesteps // self.batch_size
        self._best_eval_return = None
        agent = Agent(self.envs, goal_size=self.goal_size).to(self.device)
        optimizer = optim.Adam(agent.parameters(), lr=self.learning_rate, eps=1e-5)

        if self.load_default_checkpoint:
            ckpt_path = next(
                (p for p in (self.best_model_path, self.model_path) if os.path.exists(p)),
                None,
            )
            if ckpt_path is not None:
                agent.load_state_dict(torch.load(ckpt_path, map_location=self.device))
                print(f"[checkpoint] loaded default checkpoint: {ckpt_path}")
            else:
                print(
                    f"[checkpoint] load_default_checkpoint=True but no checkpoint "
                    f"found under {self.run_dir}; training from scratch"
                )
        else:
            print("[checkpoint] load_default_checkpoint=False; training from scratch")

        obs = torch.zeros((self.num_steps, self.num_envs, self.envs.policy_obs_dim)).to(self.device)
        actions = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        real_next_values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        action_low = self.envs.action_low
        action_high = self.envs.action_high

        global_step = 0
        global_episodes = 0
        last_save_step = 0
        last_eval_step = 0
        start_time = time.time()
        next_obs = self.envs.reset(seed=self.seed)
        next_done = torch.zeros(self.num_envs).to(self.device)

        num_params = sum(p.numel() for p in agent.parameters())
        self.writer.add_scalar("charts/num_parameters", num_params, 0)
        self.writer.add_scalar("charts/num_iterations", self.num_iterations, 0)

        for iteration in range(1, self.num_iterations + 1):
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            iter_ep_returns, iter_ep_lengths = [], []
            iter_terminations, iter_truncations = 0, 0
            action_clip_fracs = []
            rollout_start = time.time()

            for step in range(0, self.num_steps):
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                clipped_action = torch.clamp(action, action_low, action_high)
                action_clip_fracs.append((clipped_action != action).float().mean().item())

                next_obs, reward, terminations, truncations, infos = self.envs.step(clipped_action)
                done = terminations | truncations
                iter_terminations += int(terminations.sum().item())
                iter_truncations += int((truncations & ~terminations).sum().item())
                rewards[step] = reward

                with torch.no_grad():
                    real_next_values[step] = (
                        agent.get_value(infos["final_obs"]).flatten()
                        * (1.0 - terminations.float())
                    )
                next_done = done.float()

                if bool(done.any()):
                    ep_returns = infos["episodic_return"][done]
                    ep_lengths = infos["episodic_length"][done]
                    for ep_r, ep_l in zip(ep_returns.tolist(), ep_lengths.tolist()):
                        self.writer.add_scalar("charts/episodic_return", ep_r, global_step)
                        self.writer.add_scalar("charts/episodic_length", ep_l, global_step)
                        iter_ep_returns.append(float(ep_r))
                        iter_ep_lengths.append(float(ep_l))

            rollout_time = time.time() - rollout_start
            global_episodes += len(iter_ep_returns)
            update_start = time.time()

            with torch.no_grad():
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                    delta = rewards[t] + self.gamma * real_next_values[t] - values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            b_obs = obs.reshape((-1, self.envs.policy_obs_dim))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(self.batch_size)
            clipfracs = []
            grad_norms = []
            ratio_mins, ratio_maxs = [], []
            pg_losses, v_losses, entropy_losses, total_losses = [], [], [], []
            epochs_run = 0
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                epoch_kls = []
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
                        epoch_kls.append(approx_kl.item())
                        ratio_mins.append(ratio.min().item())
                        ratio_maxs.append(ratio.max().item())

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
                    optimizer.step()

                    grad_norms.append(float(grad_norm))
                    pg_losses.append(pg_loss.item())
                    v_losses.append(v_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    total_losses.append(loss.item())

                epochs_run += 1
                if self.target_kl is not None and np.mean(epoch_kls) > self.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            now = time.time()
            sps = int(global_step / (now - start_time))
            update_time = now - update_start
            iteration_time = now - rollout_start
            iteration_fps = self.batch_size / max(iteration_time, 1e-8)
            rollout_fps = self.batch_size / max(rollout_time, 1e-8)
            update_fps = self.batch_size / max(update_time, 1e-8)

            self.writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, global_step)

            self.writer.add_scalar("losses/total_loss", float(np.mean(total_losses)), global_step)
            self.writer.add_scalar("losses/policy_loss_mean", float(np.mean(pg_losses)), global_step)
            self.writer.add_scalar("losses/policy_loss_std", float(np.std(pg_losses)), global_step)
            self.writer.add_scalar("losses/value_loss_mean", float(np.mean(v_losses)), global_step)
            self.writer.add_scalar("losses/value_loss_std", float(np.std(v_losses)), global_step)
            self.writer.add_scalar("losses/entropy_mean", float(np.mean(entropy_losses)), global_step)
            self.writer.add_scalar("losses/clipfrac_max", float(np.max(clipfracs)), global_step)
            self.writer.add_scalar("losses/grad_norm", float(np.mean(grad_norms)), global_step)
            self.writer.add_scalar("losses/grad_norm_max", float(np.max(grad_norms)), global_step)
            self.writer.add_scalar("losses/ratio_min", float(np.min(ratio_mins)), global_step)
            self.writer.add_scalar("losses/ratio_max", float(np.max(ratio_maxs)), global_step)
            self.writer.add_scalar("charts/epochs_run", epochs_run, global_step)

            self.writer.add_scalar("charts/SPS", sps, global_step)
            self.writer.add_scalar("charts/FPS", iteration_fps, global_step)
            self.writer.add_scalar("charts/iteration", iteration, global_step)
            self.writer.add_scalar("charts/global_step", global_step, global_step)
            self.writer.add_scalar("charts/total_episodes", global_episodes, global_step)
            self.writer.add_scalar("time/rollout_seconds", rollout_time, global_step)
            self.writer.add_scalar("time/update_seconds", update_time, global_step)
            self.writer.add_scalar("time/iteration_seconds", iteration_time, global_step)
            self.writer.add_scalar("time/elapsed_seconds", now - start_time, global_step)
            self.writer.add_scalar("time/rollout_fps", rollout_fps, global_step)
            self.writer.add_scalar("time/update_fps", update_fps, global_step)
            self.writer.add_scalar("time/iteration_fps", iteration_fps, global_step)

            done_total = iter_terminations + iter_truncations
            self.writer.add_scalar("charts/episodes_this_iter", len(iter_ep_returns), global_step)
            self.writer.add_scalar("charts/terminations", iter_terminations, global_step)
            self.writer.add_scalar("charts/truncations", iter_truncations, global_step)
            self.writer.add_scalar(
                "charts/termination_fraction",
                iter_terminations / max(done_total, 1),
                global_step,
            )
            if len(iter_ep_returns) > 0:
                ep_r = np.array(iter_ep_returns)
                ep_l = np.array(iter_ep_lengths)
                self.writer.add_scalar("charts/episodic_return_mean", float(ep_r.mean()), global_step)
                self.writer.add_scalar("charts/episodic_return_std", float(ep_r.std()), global_step)
                self.writer.add_scalar("charts/episodic_return_min", float(ep_r.min()), global_step)
                self.writer.add_scalar("charts/episodic_return_max", float(ep_r.max()), global_step)
                self.writer.add_scalar("charts/episodic_length_mean", float(ep_l.mean()), global_step)
                self.writer.add_scalar("charts/episodic_length_min", float(ep_l.min()), global_step)
                self.writer.add_scalar("charts/episodic_length_max", float(ep_l.max()), global_step)
                self.writer.add_histogram("rollout/episodic_return_hist", ep_r, global_step)

            self.writer.add_scalar("rollout/reward_mean", rewards.mean().item(), global_step)
            self.writer.add_scalar("rollout/reward_std", rewards.std().item(), global_step)
            self.writer.add_scalar("rollout/reward_max", rewards.max().item(), global_step)
            self.writer.add_scalar("rollout/reward_per_env", rewards.sum().item() / self.num_envs, global_step)
            self.writer.add_scalar("rollout/advantage_mean", b_advantages.mean().item(), global_step)
            self.writer.add_scalar("rollout/advantage_std", b_advantages.std().item(), global_step)
            self.writer.add_scalar("rollout/advantage_abs_mean", b_advantages.abs().mean().item(), global_step)
            self.writer.add_scalar("rollout/return_mean", b_returns.mean().item(), global_step)
            self.writer.add_scalar("rollout/return_std", b_returns.std().item(), global_step)
            self.writer.add_scalar("rollout/value_mean", b_values.mean().item(), global_step)
            self.writer.add_scalar("rollout/value_std", b_values.std().item(), global_step)
            self.writer.add_scalar("rollout/logprob_mean", b_logprobs.mean().item(), global_step)
            self.writer.add_scalar("rollout/logprob_std", b_logprobs.std().item(), global_step)
            self.writer.add_scalar("rollout/action_clip_fraction", float(np.mean(action_clip_fracs)), global_step)
            self.writer.add_scalar("rollout/done_fraction", dones.mean().item(), global_step)

            action_std = agent.actor_logstd.detach().exp().flatten()
            self.writer.add_scalar("policy/action_std_mean", action_std.mean().item(), global_step)
            self.writer.add_scalar("policy/action_std_min", action_std.min().item(), global_step)
            self.writer.add_scalar("policy/action_std_max", action_std.max().item(), global_step)
            for dim, std in enumerate(action_std.tolist()):
                self.writer.add_scalar(f"policy/action_std/dim_{dim}", std, global_step)
            self.writer.add_scalar("policy/action_mean", b_actions.mean().item(), global_step)
            self.writer.add_scalar("policy/action_abs_mean", b_actions.abs().mean().item(), global_step)
            weight_norm = torch.sqrt(sum((p.detach() ** 2).sum() for p in agent.parameters()))
            self.writer.add_scalar("policy/weight_norm", weight_norm.item(), global_step)

            self.writer.add_histogram("rollout/actions", b_actions, global_step)
            self.writer.add_histogram("rollout/advantages", b_advantages, global_step)
            self.writer.add_histogram("rollout/returns", b_returns, global_step)
            self.writer.add_histogram("rollout/values", b_values, global_step)
            self.writer.add_histogram("rollout/logprobs", b_logprobs, global_step)
            self.writer.add_histogram("rollout/rewards", rewards, global_step)
            for name, param in agent.named_parameters():
                self.writer.add_histogram(f"weights/{name}", param.detach(), global_step)
                if param.grad is not None:
                    self.writer.add_histogram(f"grads/{name}", param.grad.detach(), global_step)

            if save_model and save_freq > 0 and global_step - last_save_step >= save_freq:
                last_save_step = global_step
                ckpt_path = os.path.join(
                    self.run_dir, f"{self.exp_name}_step{global_step:09d}.cleanrl_model"
                )
                torch.save(agent.state_dict(), ckpt_path)
                print(
                    f"[checkpoint] global_step={global_step}/{total_timesteps}  ->  {ckpt_path}"
                )

            if save_model and global_step - last_eval_step >= eval_freq:
                last_eval_step = global_step
                self._evaluate_agent(agent, global_step)

        if save_model:
            self._evaluate_agent(agent, global_step)

        self.envs.close()
        self.writer.close()

        if self.track:
            import wandb

            wandb.finish()

