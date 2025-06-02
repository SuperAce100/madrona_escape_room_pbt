import torch
from torch import nn
import torch.nn.functional as F
import torch._dynamo
from torch import optim
from torch.func import vmap
from os import environ as env_vars
from typing import Callable, Optional
from dataclasses import dataclass, field
from typing import List, Dict
from .profile import profile
from time import time
from pathlib import Path
import math  # Use math instead of numpy for argmax
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime
import numpy as np

from .cfg import TrainConfig, SimInterface, PPOConfig
from .rollouts import RolloutManager, Rollouts
from .amp import AMPState
from .actor_critic import ActorCritic
from .moving_avg import EMANormalizer
from .learning_state import LearningState


@dataclass(frozen=True)
class MiniBatch:
    obs: List[torch.Tensor]
    actions: torch.Tensor
    log_probs: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    rnn_start_states: tuple[torch.Tensor, ...]


@dataclass
class PPOStats:
    loss: float = 0
    action_loss: float = 0
    value_loss: float = 0
    entropy_loss: float = 0
    returns_mean: float = 0
    returns_stddev: float = 0


@dataclass(frozen=True)
class UpdateResult:
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    bootstrap_values: torch.Tensor
    ppo_stats: PPOStats


@dataclass
class ParallelPolicyState:
    policies: List[ActorCritic]
    optimizers: List[torch.optim.Optimizer]
    value_normalizers: List[EMANormalizer]
    rollout_managers: List[RolloutManager]
    best_policy_idx: int = 0
    policy_returns: List[float] = field(default_factory=list)
    best_policy_returns: List[float] = field(default_factory=list)
    best_policy_stats: List[PPOStats] = field(default_factory=list)
    schedulers: List[torch.optim.lr_scheduler.LRScheduler] = field(default_factory=list)


def _mb_slice(tensor, inds):
    # Tensors come from the rollout manager as (C, T, N, ...)
    # Want to select mb from C * N and keep sequences of length T

    return tensor.transpose(0, 1).reshape(
        tensor.shape[1], tensor.shape[0] * tensor.shape[2], *tensor.shape[3:]
    )[:, inds, ...]


def _mb_slice_rnn(rnn_state, inds):
    # RNN state comes from the rollout manager as (C, :, :, N, :)
    # Want to select minibatch from C * N and keep sequences of length T

    reshaped = rnn_state.permute(1, 2, 0, 3, 4).reshape(
        rnn_state.shape[1], rnn_state.shape[2], -1, rnn_state.shape[4]
    )

    return reshaped[:, :, inds, :]


def _gather_minibatch(
    rollouts: Rollouts, advantages: torch.Tensor, inds: torch.Tensor, amp: AMPState
):
    obs_slice = list(_mb_slice(obs, inds) for obs in rollouts.obs)

    actions_slice = _mb_slice(rollouts.actions, inds)
    log_probs_slice = _mb_slice(rollouts.log_probs, inds).to(dtype=amp.compute_dtype)
    dones_slice = _mb_slice(rollouts.dones, inds)
    rewards_slice = _mb_slice(rollouts.rewards, inds).to(dtype=amp.compute_dtype)
    values_slice = _mb_slice(rollouts.values, inds).to(dtype=amp.compute_dtype)
    advantages_slice = _mb_slice(advantages, inds).to(dtype=amp.compute_dtype)

    rnn_starts_slice = tuple(
        _mb_slice_rnn(state, inds) for state in rollouts.rnn_start_states
    )

    return MiniBatch(
        obs=obs_slice,
        actions=actions_slice,
        log_probs=log_probs_slice,
        dones=dones_slice,
        rewards=rewards_slice,
        values=values_slice,
        advantages=advantages_slice,
        rnn_start_states=rnn_starts_slice,
    )


def _compute_advantages(
    cfg: TrainConfig,
    amp: AMPState,
    value_normalizer: EMANormalizer,
    advantages_out: torch.Tensor,
    rollouts: Rollouts,
):
    num_chunks, steps_per_chunk, N = rollouts.dones.shape[0:3]
    T = num_chunks * steps_per_chunk

    seq_dones = rollouts.dones.view(T, N, 1)
    seq_rewards = rollouts.rewards.view(T, N, 1)
    seq_values = rollouts.values.view(T, N, 1)
    seq_advantages_out = advantages_out.view(T, N, 1)

    next_advantage = 0.0
    next_values = rollouts.bootstrap_values

    # Compute advantages with GAE
    for i in reversed(range(cfg.steps_per_update)):
        cur_dones = seq_dones[i].to(dtype=amp.compute_dtype)
        cur_rewards = seq_rewards[i].to(dtype=amp.compute_dtype)
        cur_values = seq_values[i].to(dtype=amp.compute_dtype)

        next_valid = 1.0 - cur_dones

        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        td_err = cur_rewards + cfg.gamma * next_valid * next_values - cur_values

        # A_t = sum (gamma * lambda)^(l - 1) * delta_l (EQ 16 GAE)
        #     = delta_t + gamma * lambda * A_t+1
        cur_advantage = (
            td_err + cfg.gamma * cfg.gae_lambda * next_valid * next_advantage
        )

        seq_advantages_out[i] = cur_advantage

        next_advantage = cur_advantage
        next_values = cur_values

    # Normalize advantages
    if cfg.normalize_advantages:
        with torch.no_grad():
            advantages_mean = advantages_out.mean()
            advantages_std = advantages_out.std().clamp(min=1e-8)
            advantages_out.sub_(advantages_mean).div_(advantages_std)


def _compute_action_scores(cfg, amp, advantages):
    if not cfg.normalize_advantages:
        return advantages
    else:
        # Unclear from docs if var_mean is safe under autocast
        with amp.disable():
            var, mean = torch.var_mean(advantages.to(dtype=torch.float32))
            action_scores = advantages - mean
            action_scores.mul_(torch.rsqrt(var.clamp(min=1e-5)))

            return action_scores.to(dtype=amp.compute_dtype)


def _ppo_update(
    cfg: TrainConfig,
    amp: AMPState,
    mb: MiniBatch,
    actor_critic: ActorCritic,
    optimizer: torch.optim.Optimizer,
    value_normalizer: EMANormalizer,
):
    with amp.enable():
        # Forward pass for actor and critic
        with profile("AC Forward", gpu=True):
            new_log_probs, entropies, new_values = actor_critic.fwd_update(
                mb.rnn_start_states, mb.dones, mb.actions, *mb.obs
            )

        # Compute PPO loss
        with torch.no_grad():
            ratio = torch.exp(new_log_probs - mb.log_probs)
            surr1 = mb.advantages * ratio
            surr2 = mb.advantages * torch.clamp(
                ratio, 1.0 - cfg.ppo.clip_coef, 1.0 + cfg.ppo.clip_coef
            )
            action_obj = torch.min(surr1, surr2)

        # Compute value loss with proper normalization
        returns = mb.advantages + mb.values
        normalized_returns = value_normalizer(amp, returns)

        if cfg.ppo.clip_value_loss:
            value_pred_clipped = mb.values + (new_values - mb.values).clamp(
                -cfg.ppo.clip_coef, cfg.ppo.clip_coef
            )
            value_losses = (new_values - normalized_returns).pow(2)
            value_losses_clipped = (value_pred_clipped - normalized_returns).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * F.mse_loss(new_values, normalized_returns)

        # Compute entropy loss with proper scaling
        entropy_loss = -entropies.mean()

        # Total loss with proper weighting
        loss = (
            -action_obj.mean()
            + cfg.ppo.value_loss_coef * value_loss
            + cfg.ppo.entropy_coef * entropy_loss
        )

    # Optimize with gradient clipping
    with profile("Optimize"):
        if amp.scaler is None:
            loss.backward()
            nn.utils.clip_grad_norm_(actor_critic.parameters(), cfg.ppo.max_grad_norm)
            optimizer.step()
        else:
            amp.scaler.scale(loss).backward()
            amp.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(actor_critic.parameters(), cfg.ppo.max_grad_norm)
            amp.scaler.step(optimizer)
            amp.scaler.update()

        optimizer.zero_grad()

    # Compute statistics
    with torch.no_grad():
        returns_var, returns_mean = torch.var_mean(normalized_returns)
        returns_stddev = torch.sqrt(returns_var)

        stats = PPOStats(
            loss=loss.cpu().float().item(),
            action_loss=-(action_obj.mean().cpu().float().item()),
            value_loss=value_loss.cpu().float().item(),
            entropy_loss=-(entropies.mean().cpu().float().item()),
            returns_mean=returns_mean.cpu().float().item(),
            returns_stddev=returns_stddev.cpu().float().item(),
        )

    return stats


def _update_iter(
    cfg: TrainConfig,
    amp: AMPState,
    num_train_seqs: int,
    sim: SimInterface,
    rollout_mgr: RolloutManager,
    advantages: torch.Tensor,
    parallel_state: ParallelPolicyState,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    update_idx: int,
    writer: Optional[SummaryWriter] = None,
):
    with torch.no_grad():
        # Evaluate all policies in parallel
        all_rollouts = []
        all_advantages = []
        all_returns = []
        all_max_ys = []  # Track max y positions for each policy

        for i, (policy, value_normalizer, rollout_mgr) in enumerate(
            zip(
                parallel_state.policies,
                parallel_state.value_normalizers,
                parallel_state.rollout_managers,
            )
        ):
            policy.eval()
            value_normalizer.eval()

            # Create policy-specific config
            policy_cfg = TrainConfig(
                num_updates=cfg.num_updates,
                steps_per_update=cfg.steps_per_update,
                num_bptt_chunks=cfg.num_bptt_chunks,
                lr=policy.hyperparams["lr"],
                gamma=policy.hyperparams["gamma"],
                gae_lambda=cfg.gae_lambda,
                ppo=PPOConfig(
                    num_mini_batches=cfg.ppo.num_mini_batches,
                    clip_coef=cfg.ppo.clip_coef,
                    value_loss_coef=policy.hyperparams["value_loss_coef"],
                    entropy_coef=policy.hyperparams["entropy_coef"],
                    max_grad_norm=cfg.ppo.max_grad_norm,
                    num_epochs=cfg.ppo.num_epochs,
                    clip_value_loss=cfg.ppo.clip_value_loss,
                ),
                value_normalizer_decay=cfg.value_normalizer_decay,
                mixed_precision=cfg.mixed_precision,
            )

            with profile("Collect Rollouts"):
                rollouts = rollout_mgr.collect(amp, sim, policy, value_normalizer)
                all_rollouts.append(rollouts)

            with profile("Compute Advantages"):
                policy_advantages = torch.zeros_like(rollouts.rewards)
                _compute_advantages(
                    policy_cfg, amp, value_normalizer, policy_advantages, rollouts
                )
                all_advantages.append(policy_advantages)

            # Calculate returns and max y position for this policy
            returns = rollouts.rewards.sum().item()
            max_y = rollouts.rewards.view(-1, rollouts.rewards.shape[2]).max(dim=0)[0].mean().item()
            all_returns.append(returns)
            all_max_ys.append(max_y)

            # Update learning rate based on returns
            if scheduler is not None:
                scheduler.step(returns)

        # Select best policy based on progress through rooms
        best_idx = max(range(len(all_max_ys)), key=lambda i: all_max_ys[i])
        parallel_state.best_policy_idx = best_idx
        parallel_state.policy_returns = all_returns

        # Track best policy's returns and progress
        parallel_state.best_policy_returns.append(all_returns[best_idx])

        # Log metrics for each policy if writer is provided
        if writer is not None:
            for i, (ret, max_y) in enumerate(zip(all_returns, all_max_ys)):
                writer.add_scalar(f"policy_{i}/returns", ret, update_idx)
                writer.add_scalar(f"policy_{i}/max_y", max_y, update_idx)
                
                # Log room success rates for each policy
                room1_success = (max_y >= 13.33).float().mean().item()
                room2_success = (max_y >= 26.67).float().mean().item()
                room3_success = (max_y >= 40.0).float().mean().item()
                
                writer.add_scalar(f"policy_{i}/room1_success", room1_success, update_idx)
                writer.add_scalar(f"policy_{i}/room2_success", room2_success, update_idx)
                writer.add_scalar(f"policy_{i}/room3_success", room3_success, update_idx)
                
                # Log progress metrics
                progress = max_y / 40.0  # Normalized progress
                writer.add_scalar(f"policy_{i}/progress", progress, update_idx)
                
                # Log hyperparameters
                policy = parallel_state.policies[i]
                writer.add_scalar(f"policy_{i}/lr", policy.hyperparams["lr"], update_idx)
                writer.add_scalar(f"policy_{i}/gamma", policy.hyperparams["gamma"], update_idx)
                writer.add_scalar(f"policy_{i}/entropy_coef", policy.hyperparams["entropy_coef"], update_idx)
                writer.add_scalar(f"policy_{i}/value_loss_coef", policy.hyperparams["value_loss_coef"], update_idx)

            # Log best policy metrics
            writer.add_scalar("best_policy_idx", best_idx, update_idx)
            writer.add_scalar("best_policy/returns", all_returns[best_idx], update_idx)
            writer.add_scalar("best_policy/max_y", all_max_ys[best_idx], update_idx)
            writer.add_scalar("best_policy/progress", all_max_ys[best_idx] / 40.0, update_idx)

    # Train each policy using its own rollouts and advantages
    for i, (policy, optimizer, value_normalizer, rollouts, advantages) in enumerate(
        zip(
            parallel_state.policies,
            parallel_state.optimizers,
            parallel_state.value_normalizers,
            all_rollouts,
            all_advantages,
        )
    ):
        policy.train()
        value_normalizer.train()

        with profile("PPO"):
            aggregate_stats = PPOStats()
            num_stats = 0

            # Create policy-specific config
            policy_cfg = TrainConfig(
                num_updates=cfg.num_updates,
                steps_per_update=cfg.steps_per_update,
                num_bptt_chunks=cfg.num_bptt_chunks,
                lr=policy.hyperparams["lr"],
                gamma=policy.hyperparams["gamma"],
                gae_lambda=cfg.gae_lambda,
                ppo=PPOConfig(
                    num_mini_batches=cfg.ppo.num_mini_batches,
                    clip_coef=cfg.ppo.clip_coef,
                    value_loss_coef=policy.hyperparams["value_loss_coef"],
                    entropy_coef=policy.hyperparams["entropy_coef"],
                    max_grad_norm=cfg.ppo.max_grad_norm,
                    num_epochs=cfg.ppo.num_epochs,
                    clip_value_loss=cfg.ppo.clip_value_loss,
                ),
                value_normalizer_decay=cfg.value_normalizer_decay,
                mixed_precision=cfg.mixed_precision,
            )

            # Use policy's own rollouts and advantages
            for epoch in range(policy_cfg.ppo.num_epochs):
                for inds in torch.randperm(num_train_seqs).chunk(
                    policy_cfg.ppo.num_mini_batches
                ):
                    with torch.no_grad(), profile("Gather Minibatch", gpu=True):
                        mb = _gather_minibatch(rollouts, advantages, inds, amp)
                    cur_stats = _ppo_update(
                        policy_cfg, amp, mb, policy, optimizer, value_normalizer
                    )

                    with torch.no_grad():
                        num_stats += 1
                        aggregate_stats.loss += (
                            cur_stats.loss - aggregate_stats.loss
                        ) / num_stats
                        aggregate_stats.action_loss += (
                            cur_stats.action_loss - aggregate_stats.action_loss
                        ) / num_stats
                        aggregate_stats.value_loss += (
                            cur_stats.value_loss - aggregate_stats.value_loss
                        ) / num_stats
                        aggregate_stats.entropy_loss += (
                            cur_stats.entropy_loss - aggregate_stats.entropy_loss
                        ) / num_stats
                        aggregate_stats.returns_mean += (
                            cur_stats.returns_mean - aggregate_stats.returns_mean
                        ) / num_stats
                        aggregate_stats.returns_stddev += (
                            cur_stats.returns_stddev - aggregate_stats.returns_stddev
                        ) / num_stats

            # Track best policy's stats
            if i == best_idx:
                parallel_state.best_policy_stats.append(aggregate_stats)

            # Log training metrics for each policy if writer is provided
            if writer is not None:
                writer.add_scalar(f"policy_{i}/loss", aggregate_stats.loss, update_idx)
                writer.add_scalar(
                    f"policy_{i}/action_loss", aggregate_stats.action_loss, update_idx
                )
                writer.add_scalar(
                    f"policy_{i}/value_loss", aggregate_stats.value_loss, update_idx
                )
                writer.add_scalar(
                    f"policy_{i}/entropy_loss", aggregate_stats.entropy_loss, update_idx
                )
                writer.add_scalar(
                    f"policy_{i}/returns_mean", aggregate_stats.returns_mean, update_idx
                )
                writer.add_scalar(
                    f"policy_{i}/returns_stddev",
                    aggregate_stats.returns_stddev,
                    update_idx,
                )

                # Log best policy's stats
                if i == best_idx:
                    writer.add_scalar(
                        "best_policy/loss", aggregate_stats.loss, update_idx
                    )
                    writer.add_scalar(
                        "best_policy/action_loss",
                        aggregate_stats.action_loss,
                        update_idx,
                    )
                    writer.add_scalar(
                        "best_policy/value_loss", aggregate_stats.value_loss, update_idx
                    )
                    writer.add_scalar(
                        "best_policy/entropy_loss",
                        aggregate_stats.entropy_loss,
                        update_idx,
                    )
                    writer.add_scalar(
                        "best_policy/returns_mean",
                        aggregate_stats.returns_mean,
                        update_idx,
                    )
                    writer.add_scalar(
                        "best_policy/returns_stddev",
                        aggregate_stats.returns_stddev,
                        update_idx,
                    )

    # Return the best policy's results for logging
    return UpdateResult(
        actions=all_rollouts[best_idx].actions.view(
            -1, *all_rollouts[best_idx].actions.shape[2:]
        ),
        rewards=all_rollouts[best_idx].rewards.view(
            -1, *all_rollouts[best_idx].rewards.shape[2:]
        ),
        values=all_rollouts[best_idx].values.view(
            -1, *all_rollouts[best_idx].values.shape[2:]
        ),
        advantages=all_advantages[best_idx].view(
            -1, *all_advantages[best_idx].shape[2:]
        ),
        bootstrap_values=all_rollouts[best_idx].bootstrap_values.view(
            -1, *all_rollouts[best_idx].bootstrap_values.shape[2:]
        ),
        ppo_stats=aggregate_stats,
    )


def _update_loop(
    update_iter_fn: Callable,
    gpu_sync_fn: Callable,
    user_cb: Callable,
    cfg: TrainConfig,
    num_agents: int,
    sim: SimInterface,
    rollout_mgr: RolloutManager,
    parallel_state: ParallelPolicyState,
    start_update_idx: int,
    writer: Optional[SummaryWriter] = None,
):
    num_train_seqs = num_agents * cfg.num_bptt_chunks
    assert num_train_seqs % cfg.ppo.num_mini_batches == 0

    # Initialize amp
    amp = AMPState(rollout_mgr.dev, cfg.mixed_precision)

    for update_idx in range(start_update_idx, cfg.num_updates):
        update_start_time = time()

        with profile("Update Iter Timing"):
            update_result = update_iter_fn(
                cfg,
                amp,
                num_train_seqs,
                sim,
                rollout_mgr,
                torch.zeros_like(rollout_mgr.rewards),
                parallel_state,
                None,
                update_idx,
                writer,
            )

        profile.gpu_measure()
        profile.commit()

        update_end_time = time()
        update_time = update_end_time - update_start_time
        user_cb(update_idx, update_time, update_result, parallel_state)


def train(
    dev, sim, cfg, actor_critic, update_cb, restore_ckpt=None, num_parallel_policies=4
):
    print(cfg)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    num_agents = sim.actions.shape[0]

    # Create multiple policies
    policies = []
    optimizers = []
    value_normalizers = []
    rollout_managers = []

    # Initialize amp before creating rollout managers
    amp = AMPState(dev, cfg.mixed_precision)

    for _ in range(num_parallel_policies):
        policy = actor_critic.to(dev)
        optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
        value_normalizer = EMANormalizer(
            cfg.value_normalizer_decay, disable=not cfg.normalize_values
        )
        value_normalizer = value_normalizer.to(dev)

        # Create rollout manager for this policy
        rollout_mgr = RolloutManager(
            dev,
            sim,
            cfg.steps_per_update,
            cfg.num_bptt_chunks,
            amp,
            actor_critic.recurrent_cfg,
        )

        policies.append(policy)
        optimizers.append(optimizer)
        value_normalizers.append(value_normalizer)
        rollout_managers.append(rollout_mgr)

    parallel_state = ParallelPolicyState(
        policies=policies,
        optimizers=optimizers,
        value_normalizers=value_normalizers,
        rollout_managers=rollout_managers,
    )

    if restore_ckpt != None:
        # Load checkpoint into all policies
        for policy in policies:
            policy.load_state_dict(torch.load(restore_ckpt))
        start_update_idx = 0
    else:
        start_update_idx = 0

    rollout_mgr = RolloutManager(
        dev,
        sim,
        cfg.steps_per_update,
        cfg.num_bptt_chunks,
        amp,
        actor_critic.recurrent_cfg,
    )

    if dev.type == "cuda":

        def gpu_sync_fn():
            torch.cuda.synchronize()
    else:

        def gpu_sync_fn():
            pass

    _update_loop(
        update_iter_fn=_update_iter,
        gpu_sync_fn=gpu_sync_fn,
        user_cb=update_cb,
        cfg=cfg,
        num_agents=num_agents,
        sim=sim,
        rollout_mgr=rollout_mgr,
        parallel_state=parallel_state,
        start_update_idx=start_update_idx,
        writer=None,
    )

    # Return best policy
    return parallel_state.policies[parallel_state.best_policy_idx].cpu()


def _transfer_policy_weights(target_policy, source_policy, transfer_ratio=0.5):
    """Transfer weights from source policy to target policy with some mixing."""
    with torch.no_grad():
        for target_param, source_param in zip(target_policy.parameters(), source_policy.parameters()):
            # Mix weights with random mask
            mask = torch.rand_like(target_param) < transfer_ratio
            target_param.data = torch.where(mask, source_param.data, target_param.data)

def _tournament_select(policy_performances, tournament_size=3):
    """Select a policy using tournament selection."""
    candidates = np.random.choice(len(policy_performances), tournament_size, replace=False)
    return max(candidates, key=lambda i: policy_performances[i][1])

def _crossover_hyperparams(parent1_params, parent2_params):
    """Create new hyperparameters by crossing over two parents."""
    child_params = {}
    for key in parent1_params:
        if np.random.random() < 0.5:
            child_params[key] = parent1_params[key]
        else:
            child_params[key] = parent2_params[key]
    return child_params

def _compute_policy_score(returns_history, value_loss_history, entropy_history, window_size=10):
    """Compute a policy score based on progress through rooms."""
    if len(returns_history) < window_size:
        return np.mean(returns_history)  # Not enough history yet
    
    recent_returns = returns_history[-window_size:]
    recent_value_loss = value_loss_history[-window_size:]
    recent_entropy = entropy_history[-window_size:]
    
    # Compute metrics
    mean_return = np.mean(recent_returns)
    return_std = np.std(recent_returns)
    mean_value_loss = np.mean(recent_value_loss)
    mean_entropy = np.mean(recent_entropy)
    
    # Calculate progress through rooms
    # Assuming rewards are based on y-position progress
    progress = mean_return / 40.0  # Normalize by max possible progress
    progress_std = return_std / 40.0
    
    # Higher entropy is good (exploration), but not too high
    entropy_score = np.clip(mean_entropy, 0, 0.5) / 0.5
    
    # Lower value loss is good (better value estimation)
    value_score = np.exp(-mean_value_loss)
    
    # Progress score: higher progress with lower variance is good
    # Add small constant to denominator to handle early training
    progress_score = progress / (1 + progress_std + 1e-6)
    
    # Combine scores with focus on progress
    total_score = (
        0.6 * progress_score +    # Progress through rooms
        0.2 * value_score +      # Value function quality
        0.2 * entropy_score      # Exploration
    )
    
    return total_score

def _resample_hyperparameters(
    parallel_state: ParallelPolicyState,
    num_parallel_policies: int,
    default_lr: float,
    default_gamma: float,
    default_entropy: float,
    default_value: float,
    mutation_rate: float = 0.1,
    elite_fraction: float = 0.3,
    transfer_ratio: float = 0.5,
):
    """Resample hyperparameters based on progress through rooms."""
    num_elites = max(1, int(num_parallel_policies * elite_fraction))
    
    # Compute policy scores
    policy_scores = []
    for i in range(len(parallel_state.policies)):
        score = _compute_policy_score(
            parallel_state.policy_returns,
            [stats.value_loss for stats in parallel_state.best_policy_stats],
            [stats.entropy_loss for stats in parallel_state.best_policy_stats]
        )
        policy_scores.append((i, score))
    
    # Sort policies by their scores
    policy_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Keep elite policies unchanged
    elite_indices = [p[0] for p in policy_scores[:num_elites]]
    
    # Generate new hyperparameters and transfer weights for non-elite policies
    new_hyperparams = []
    for i in range(num_parallel_policies):
        if i in elite_indices:
            # Keep elite policy hyperparameters
            policy = parallel_state.policies[i]
            new_hyperparams.append({
                "lr": policy.hyperparams["lr"],
                "gamma": policy.hyperparams["gamma"],
                "entropy_coef": policy.hyperparams["entropy_coef"],
                "value_loss_coef": policy.hyperparams["value_loss_coef"],
            })
        else:
            # Tournament selection for parents
            parent1_idx = _tournament_select(policy_scores)
            parent2_idx = _tournament_select(policy_scores)
            
            # Crossover hyperparameters
            parent1 = parallel_state.policies[parent1_idx]
            parent2 = parallel_state.policies[parent2_idx]
            child_params = _crossover_hyperparams(parent1.hyperparams, parent2.hyperparams)
            
            # Mutate child hyperparameters with focus on exploration early
            # and exploitation later
            progress = np.mean(parallel_state.policy_returns[-10:]) / 40.0
            adaptive_mutation = mutation_rate * (1.0 - progress)  # More mutation early
            
            new_params = {
                "lr": child_params["lr"] * np.exp(np.random.normal(0, adaptive_mutation * 0.5)),
                "gamma": child_params["gamma"] * np.exp(np.random.normal(0, adaptive_mutation * 0.2)),
                "entropy_coef": child_params["entropy_coef"] * np.exp(np.random.normal(0, adaptive_mutation * 0.5)),
                "value_loss_coef": child_params["value_loss_coef"] * np.exp(np.random.normal(0, adaptive_mutation * 0.5)),
            }
            
            # Transfer weights from best parent
            best_parent_idx = parent1_idx if policy_scores[parent1_idx][1] > policy_scores[parent2_idx][1] else parent2_idx
            _transfer_policy_weights(parallel_state.policies[i], parallel_state.policies[best_parent_idx], transfer_ratio)
            
            new_hyperparams.append(new_params)
    
    # Clamp values to reasonable ranges
    for params in new_hyperparams:
        params["lr"] = np.clip(params["lr"], 1e-5, 1e-3)
        params["gamma"] = np.clip(params["gamma"], 0.95, 0.999)
        params["entropy_coef"] = np.clip(params["entropy_coef"], 0.001, 0.2)
        params["value_loss_coef"] = np.clip(params["value_loss_coef"], 0.1, 2.0)
    
    return new_hyperparams

def _update_hyperparameters(
    parallel_state: ParallelPolicyState,
    new_hyperparams: List[Dict],
    dev: torch.device,
    cfg: TrainConfig,
    sim: SimInterface,
    amp: AMPState,
):
    """Update policies with new hyperparameters while preserving their weights."""
    for i, (policy, params) in enumerate(zip(parallel_state.policies, new_hyperparams)):
        # Update hyperparameters
        policy.hyperparams = params
        policy._hyperparams = params
        
        # Update optimizer with new learning rate
        parallel_state.optimizers[i] = optim.Adam(
            policy.parameters(), 
            lr=params["lr"],
            eps=1e-5
        )
        
        # Update scheduler with new learning rate
        parallel_state.schedulers[i] = optim.lr_scheduler.OneCycleLR(
            parallel_state.optimizers[i],
            max_lr=params["lr"],
            total_steps=cfg.num_updates,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=10000.0,
        )
        
        # Update hyperparameters tensor
        policy.register_buffer(
            "_hyperparams_tensor",
            torch.tensor([
                params["lr"],
                params["gamma"],
                params["entropy_coef"],
                params["value_loss_coef"],
            ])
        )

def train_parallel(
    dev, sim, cfg, actor_critic, update_cb, restore_ckpt=None, num_parallel_policies=4,
    resample_interval=20,  # Resample less frequently
    mutation_rate=0.1,     # Smaller mutations
    elite_fraction=0.3,    # Keep more elites
    transfer_ratio=0.5,    # How much to transfer from parent policies
):
    """Train multiple policies in parallel with evolutionary hyperparameter optimization."""
    print(f"Starting parallel training with {num_parallel_policies} policies")
    print(cfg)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    num_agents = sim.actions.shape[0]
    total_worlds = sim.actions.shape[0] // 2

    # Initialize policies with random hyperparameters
    policies = []
    optimizers = []
    value_normalizers = []
    rollout_managers = []
    schedulers = []

    # Default hyperparameters
    default_lr = cfg.lr
    default_gamma = cfg.gamma
    default_entropy = cfg.ppo.entropy_coef
    default_value = cfg.ppo.value_loss_coef

    # Initialize amp
    amp = AMPState(dev, cfg.mixed_precision)

    # Generate initial random hyperparameters with smaller variance
    policy_lrs = np.exp(np.random.normal(0, 0.2, num_parallel_policies)) * default_lr
    policy_gammas = np.exp(np.random.normal(0, 0.1, num_parallel_policies)) * default_gamma
    policy_entropies = np.exp(np.random.normal(0, 0.2, num_parallel_policies)) * default_entropy
    policy_values = np.exp(np.random.normal(0, 0.2, num_parallel_policies)) * default_value

    # Clamp initial values to reasonable ranges
    policy_lrs = np.clip(policy_lrs, 1e-5, 1e-3)
    policy_gammas = np.clip(policy_gammas, 0.95, 0.999)
    policy_entropies = np.clip(policy_entropies, 0.001, 0.2)
    policy_values = np.clip(policy_values, 0.1, 2.0)

    # Create initial policies
    for i in range(num_parallel_policies):
        policy = actor_critic.to(dev)
        
        # Initialize weights with smaller variance
        for param in policy.parameters():
            if len(param.shape) > 1:
                torch.nn.init.xavier_uniform_(
                    param, gain=math.exp(torch.randn(1).item() * 0.1)
                )

        optimizer = optim.Adam(policy.parameters(), lr=policy_lrs[i], eps=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=policy_lrs[i],
            total_steps=cfg.num_updates,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=10000.0,
        )

        value_normalizer = EMANormalizer(
            cfg.value_normalizer_decay, disable=not cfg.normalize_values
        ).to(dev)

        rollout_mgr = RolloutManager(
            dev, sim, cfg.steps_per_update, cfg.num_bptt_chunks,
            amp, actor_critic.recurrent_cfg
        )

        policy.hyperparams = {
            "lr": float(policy_lrs[i]),
            "gamma": float(policy_gammas[i]),
            "entropy_coef": float(policy_entropies[i]),
            "value_loss_coef": float(policy_values[i]),
            "total_worlds": total_worlds,
        }
        policy._hyperparams = policy.hyperparams

        policy.register_buffer(
            "_hyperparams_tensor",
            torch.tensor([
                policy.hyperparams["lr"],
                policy.hyperparams["gamma"],
                policy.hyperparams["entropy_coef"],
                policy.hyperparams["value_loss_coef"],
            ])
        )

        policies.append(policy)
        optimizers.append(optimizer)
        value_normalizers.append(value_normalizer)
        rollout_managers.append(rollout_mgr)
        schedulers.append(scheduler)

    parallel_state = ParallelPolicyState(
        policies=policies,
        optimizers=optimizers,
        value_normalizers=value_normalizers,
        rollout_managers=rollout_managers,
    )
    parallel_state.schedulers = schedulers

    if restore_ckpt is not None:
        for policy in policies:
            policy.load_state_dict(torch.load(restore_ckpt))
        start_update_idx = 0
    else:
        start_update_idx = 0

    def gpu_sync_fn():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    def update_iter_wrapper(
        cfg, amp, num_train_seqs, sim, rollout_mgr, advantages,
        parallel_state, scheduler, update_idx, writer
    ):
        # Resample hyperparameters periodically
        if update_idx > 0 and update_idx % resample_interval == 0:
            new_hyperparams = _resample_hyperparameters(
                parallel_state,
                num_parallel_policies,
                default_lr,
                default_gamma,
                default_entropy,
                default_value,
                mutation_rate,
                elite_fraction,
                transfer_ratio
            )
            _update_hyperparameters(
                parallel_state,
                new_hyperparams,
                dev,
                cfg,
                sim,
                amp
            )

        return _update_iter(
            cfg, amp, num_train_seqs, sim, rollout_mgr,
            advantages, parallel_state, scheduler, update_idx, writer
        )

    _update_loop(
        update_iter_fn=update_iter_wrapper,
        gpu_sync_fn=gpu_sync_fn,
        user_cb=update_cb,
        cfg=cfg,
        num_agents=num_agents,
        sim=sim,
        rollout_mgr=rollout_managers[0],
        parallel_state=parallel_state,
        start_update_idx=start_update_idx,
        writer=None,
    )

    return parallel_state.policies[parallel_state.best_policy_idx]
