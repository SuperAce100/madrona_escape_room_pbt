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

from .cfg import TrainConfig, SimInterface, PPOConfig
from .rollouts import RolloutManager, Rollouts
from .amp import AMPState
from .actor_critic import ActorCritic
from .moving_avg import EMANormalizer
from .learning_state import LearningState

@dataclass(frozen = True)
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
    loss : float = 0
    action_loss : float = 0
    value_loss : float = 0
    entropy_loss : float = 0
    returns_mean : float = 0
    returns_stddev : float = 0


@dataclass(frozen = True)
class UpdateResult:
    actions : torch.Tensor
    rewards : torch.Tensor
    values : torch.Tensor
    advantages : torch.Tensor
    bootstrap_values : torch.Tensor
    ppo_stats : PPOStats


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


def _mb_slice(tensor, inds):
    # Tensors come from the rollout manager as (C, T, N, ...)
    # Want to select mb from C * N and keep sequences of length T

    return tensor.transpose(0, 1).reshape(
        tensor.shape[1], tensor.shape[0] * tensor.shape[2], *tensor.shape[3:])[:, inds, ...]

def _mb_slice_rnn(rnn_state, inds):
    # RNN state comes from the rollout manager as (C, :, :, N, :)
    # Want to select minibatch from C * N and keep sequences of length T

    reshaped = rnn_state.permute(1, 2, 0, 3, 4).reshape(
        rnn_state.shape[1], rnn_state.shape[2], -1, rnn_state.shape[4])

    return reshaped[:, :, inds, :] 

def _gather_minibatch(rollouts : Rollouts,
                      advantages : torch.Tensor,
                      inds : torch.Tensor,
                      amp : AMPState):
    obs_slice = list(_mb_slice(obs, inds) for obs in rollouts.obs)
    
    actions_slice = _mb_slice(rollouts.actions, inds)
    log_probs_slice = _mb_slice(rollouts.log_probs, inds).to(
        dtype=amp.compute_dtype)
    dones_slice = _mb_slice(rollouts.dones, inds)
    rewards_slice = _mb_slice(rollouts.rewards, inds).to(
        dtype=amp.compute_dtype)
    values_slice = _mb_slice(rollouts.values, inds).to(
        dtype=amp.compute_dtype)
    advantages_slice = _mb_slice(advantages, inds).to(
        dtype=amp.compute_dtype)

    rnn_starts_slice = tuple(
        _mb_slice_rnn(state, inds) for state in rollouts.rnn_start_states)

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

def _compute_advantages(cfg : TrainConfig,
                        amp : AMPState,
                        value_normalizer : EMANormalizer,
                        advantages_out : torch.Tensor,
                        rollouts : Rollouts):
    # This function is going to be operating in fp16 mode completely
    # when mixed precision is enabled since amp.compute_dtype is fp16
    # even though there is no autocast here. Unclear if this is desirable or
    # even beneficial for performance.

    num_chunks, steps_per_chunk, N = rollouts.dones.shape[0:3]
    T = num_chunks * steps_per_chunk

    seq_dones = rollouts.dones.view(T, N, 1)
    seq_rewards = rollouts.rewards.view(T, N, 1)
    seq_values = rollouts.values.view(T, N, 1)
    seq_advantages_out = advantages_out.view(T, N, 1)

    next_advantage = 0.0
    next_values = rollouts.bootstrap_values
    for i in reversed(range(cfg.steps_per_update)):
        cur_dones = seq_dones[i].to(dtype=amp.compute_dtype)
        cur_rewards = seq_rewards[i].to(dtype=amp.compute_dtype)
        cur_values = seq_values[i].to(dtype=amp.compute_dtype)

        next_valid = 1.0 - cur_dones

        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        td_err = (cur_rewards + 
            cfg.gamma * next_valid * next_values - cur_values)

        # A_t = sum (gamma * lambda)^(l - 1) * delta_l (EQ 16 GAE)
        #     = delta_t + gamma * lambda * A_t+1
        cur_advantage = (td_err +
            cfg.gamma * cfg.gae_lambda * next_valid * next_advantage)

        seq_advantages_out[i] = cur_advantage

        next_advantage = cur_advantage
        next_values = cur_values

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

def _ppo_update(cfg : TrainConfig,
                amp : AMPState,
                mb : MiniBatch,
                actor_critic : ActorCritic,
                optimizer : torch.optim.Optimizer,
                value_normalizer : EMANormalizer,
            ):
    with amp.enable():
        # Parallel forward pass for actor and critic
        with profile('AC Forward', gpu=True):
            new_log_probs, entropies, new_values = actor_critic.fwd_update(
                mb.rnn_start_states, mb.dones, mb.actions, *mb.obs)

        # Vectorized advantage computation
        with torch.no_grad():
            action_scores = _compute_action_scores(cfg, amp, mb.advantages)
            ratio = torch.exp(new_log_probs - mb.log_probs)
            surr1 = action_scores * ratio
            surr2 = action_scores * (
                torch.clamp(ratio, 1.0 - cfg.ppo.clip_coef, 1.0 + cfg.ppo.clip_coef))

        # Parallel loss computation
        action_obj = torch.min(surr1, surr2)
        returns = mb.advantages + mb.values

        if cfg.ppo.clip_value_loss:
            with torch.no_grad():
                low = mb.values - cfg.ppo.clip_coef
                high = mb.values + cfg.ppo.clip_coef
            new_values = torch.clamp(new_values, low, high)

        # Vectorized value normalization and loss
        normalized_returns = value_normalizer(amp, returns)
        value_loss = 0.5 * F.mse_loss(
            new_values, normalized_returns, reduction='none')

        # Parallel reduction operations
        action_obj = torch.mean(action_obj)
        value_loss = torch.mean(value_loss)
        entropies = torch.mean(entropies)

        loss = (
            - action_obj
            + cfg.ppo.value_loss_coef * value_loss
            - cfg.ppo.entropy_coef * entropies
        )

    # Optimized backward pass
    with profile('Optimize'):
        if amp.scaler is None:
            loss.backward()
            nn.utils.clip_grad_norm_(
                actor_critic.parameters(), cfg.ppo.max_grad_norm)
            optimizer.step()
        else:
            amp.scaler.scale(loss).backward()
            amp.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                actor_critic.parameters(), cfg.ppo.max_grad_norm)
            amp.scaler.step(optimizer)
            amp.scaler.update()

        optimizer.zero_grad()

    # Parallel statistics computation
    with torch.no_grad():
        returns_var, returns_mean = torch.var_mean(normalized_returns)
        returns_stddev = torch.sqrt(returns_var)

        stats = PPOStats(
            loss = loss.cpu().float().item(),
            action_loss = -(action_obj.cpu().float().item()),
            value_loss = value_loss.cpu().float().item(),
            entropy_loss = -(entropies.cpu().float().item()),
            returns_mean = returns_mean.cpu().float().item(),
            returns_stddev = returns_stddev.cpu().float().item(),
        )

    return stats

def _update_iter(cfg : TrainConfig,
                 amp : AMPState,
                 num_train_seqs : int,
                 sim : SimInterface,
                 rollout_mgr : RolloutManager,
                 advantages : torch.Tensor,
                 parallel_state : ParallelPolicyState,
                 scheduler : torch.optim.lr_scheduler.LRScheduler,
                 update_idx : int,
                 writer : Optional[SummaryWriter] = None):
    with torch.no_grad():
        # Evaluate all policies in parallel
        all_rollouts = []
        all_advantages = []
        
        for i, (policy, value_normalizer, rollout_mgr) in enumerate(zip(
            parallel_state.policies, 
            parallel_state.value_normalizers,
            parallel_state.rollout_managers
        )):
            policy.eval()
            value_normalizer.eval()
            
            # Create policy-specific config
            policy_cfg = TrainConfig(
                num_updates=cfg.num_updates,
                steps_per_update=cfg.steps_per_update,
                num_bptt_chunks=cfg.num_bptt_chunks,
                lr=policy.hyperparams['lr'],
                gamma=policy.hyperparams['gamma'],
                gae_lambda=cfg.gae_lambda,
                ppo=PPOConfig(
                    num_mini_batches=cfg.ppo.num_mini_batches,
                    clip_coef=cfg.ppo.clip_coef,
                    value_loss_coef=policy.hyperparams['value_loss_coef'],
                    entropy_coef=policy.hyperparams['entropy_coef'],
                    max_grad_norm=cfg.ppo.max_grad_norm,
                    num_epochs=cfg.ppo.num_epochs,
                    clip_value_loss=cfg.ppo.clip_value_loss,
                ),
                value_normalizer_decay=cfg.value_normalizer_decay,
                mixed_precision=cfg.mixed_precision,
            )

            with profile('Collect Rollouts'):
                rollouts = rollout_mgr.collect(amp, sim, policy, value_normalizer)
                all_rollouts.append(rollouts)
                
            with profile('Compute Advantages'):
                policy_advantages = torch.zeros_like(rollouts.rewards)
                _compute_advantages(policy_cfg, amp, value_normalizer, policy_advantages, rollouts)
                all_advantages.append(policy_advantages)
        
        # Select best policy based on returns
        returns = [rollouts.rewards.sum().item() for rollouts in all_rollouts]
        best_idx = max(range(len(returns)), key=lambda i: returns[i])
        parallel_state.best_policy_idx = best_idx
        parallel_state.policy_returns = returns
        
        # Track best policy's returns
        parallel_state.best_policy_returns.append(returns[best_idx])
        
        # Log returns for each policy if writer is provided
        if writer is not None:
            for i, ret in enumerate(returns):
                writer.add_scalar(f"policy_{i}/returns", ret, update_idx)
            writer.add_scalar("best_policy_idx", best_idx, update_idx)
            writer.add_scalar("best_policy/returns", returns[best_idx], update_idx)
    
    # Use best policy's rollouts and advantages for all policies
    best_rollouts = all_rollouts[best_idx]
    best_advantages = all_advantages[best_idx]
    
    # Train each policy using best policy's rollouts and advantages
    for i, (policy, optimizer, value_normalizer) in enumerate(zip(
        parallel_state.policies, 
        parallel_state.optimizers,
        parallel_state.value_normalizers
    )):
        policy.train()
        value_normalizer.train()

        with profile('PPO'):
            aggregate_stats = PPOStats()
            num_stats = 0

            # Create policy-specific config
            policy_cfg = TrainConfig(
                num_updates=cfg.num_updates,
                steps_per_update=cfg.steps_per_update,
                num_bptt_chunks=cfg.num_bptt_chunks,
                lr=policy.hyperparams['lr'],
                gamma=policy.hyperparams['gamma'],
                gae_lambda=cfg.gae_lambda,
                ppo=PPOConfig(
                    num_mini_batches=cfg.ppo.num_mini_batches,
                    clip_coef=cfg.ppo.clip_coef,
                    value_loss_coef=policy.hyperparams['value_loss_coef'],
                    entropy_coef=policy.hyperparams['entropy_coef'],
                    max_grad_norm=cfg.ppo.max_grad_norm,
                    num_epochs=cfg.ppo.num_epochs,
                    clip_value_loss=cfg.ppo.clip_value_loss,
                ),
                value_normalizer_decay=cfg.value_normalizer_decay,
                mixed_precision=cfg.mixed_precision,
            )

            # Use best policy's rollouts and advantages for all policies
            for epoch in range(policy_cfg.ppo.num_epochs):
                for inds in torch.randperm(num_train_seqs).chunk(
                        policy_cfg.ppo.num_mini_batches):
                    with torch.no_grad(), profile('Gather Minibatch', gpu=True):
                        mb = _gather_minibatch(best_rollouts, best_advantages, inds, amp)
                    cur_stats = _ppo_update(policy_cfg,
                                          amp,
                                          mb,
                                          policy,
                                          optimizer,
                                          value_normalizer)

                    with torch.no_grad():
                        num_stats += 1
                        aggregate_stats.loss += (cur_stats.loss - aggregate_stats.loss) / num_stats
                        aggregate_stats.action_loss += (
                            cur_stats.action_loss - aggregate_stats.action_loss) / num_stats
                        aggregate_stats.value_loss += (
                            cur_stats.value_loss - aggregate_stats.value_loss) / num_stats
                        aggregate_stats.entropy_loss += (
                            cur_stats.entropy_loss - aggregate_stats.entropy_loss) / num_stats
                        aggregate_stats.returns_mean += (
                            cur_stats.returns_mean - aggregate_stats.returns_mean) / num_stats
                        aggregate_stats.returns_stddev += (
                            cur_stats.returns_stddev - aggregate_stats.returns_stddev) / num_stats
            
            # Track best policy's stats
            if i == best_idx:
                parallel_state.best_policy_stats.append(aggregate_stats)
            
            # Log training metrics for each policy if writer is provided
            if writer is not None:
                writer.add_scalar(f"policy_{i}/loss", aggregate_stats.loss, update_idx)
                writer.add_scalar(f"policy_{i}/action_loss", aggregate_stats.action_loss, update_idx)
                writer.add_scalar(f"policy_{i}/value_loss", aggregate_stats.value_loss, update_idx)
                writer.add_scalar(f"policy_{i}/entropy_loss", aggregate_stats.entropy_loss, update_idx)
                writer.add_scalar(f"policy_{i}/returns_mean", aggregate_stats.returns_mean, update_idx)
                writer.add_scalar(f"policy_{i}/returns_stddev", aggregate_stats.returns_stddev, update_idx)
                
                # Log best policy's stats
                if i == best_idx:
                    writer.add_scalar("best_policy/loss", aggregate_stats.loss, update_idx)
                    writer.add_scalar("best_policy/action_loss", aggregate_stats.action_loss, update_idx)
                    writer.add_scalar("best_policy/value_loss", aggregate_stats.value_loss, update_idx)
                    writer.add_scalar("best_policy/entropy_loss", aggregate_stats.entropy_loss, update_idx)
                    writer.add_scalar("best_policy/returns_mean", aggregate_stats.returns_mean, update_idx)
                    writer.add_scalar("best_policy/returns_stddev", aggregate_stats.returns_stddev, update_idx)

    # Return the best policy's results for logging
    return UpdateResult(
        actions = best_rollouts.actions.view(-1, *best_rollouts.actions.shape[2:]),
        rewards = best_rollouts.rewards.view(-1, *best_rollouts.rewards.shape[2:]),
        values = best_rollouts.values.view(-1, *best_rollouts.values.shape[2:]),
        advantages = best_advantages.view(-1, *best_advantages.shape[2:]),
        bootstrap_values = best_rollouts.bootstrap_values.view(-1, *best_rollouts.bootstrap_values.shape[2:]),
        ppo_stats = aggregate_stats,
    )

def _update_loop(update_iter_fn : Callable,
                 gpu_sync_fn : Callable,
                 user_cb : Callable,
                 cfg : TrainConfig,
                 num_agents: int,
                 sim : SimInterface,
                 rollout_mgr : RolloutManager,
                 parallel_state : ParallelPolicyState,
                 start_update_idx : int,
                 writer : Optional[SummaryWriter] = None):
    num_train_seqs = num_agents * cfg.num_bptt_chunks
    assert(num_train_seqs % cfg.ppo.num_mini_batches == 0)

    # Initialize amp
    amp = AMPState(rollout_mgr.dev, cfg.mixed_precision)

    for update_idx in range(start_update_idx, cfg.num_updates):
        update_start_time  = time()

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

def train(dev, sim, cfg, actor_critic, update_cb, restore_ckpt=None, num_parallel_policies=4):
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
        value_normalizer = EMANormalizer(cfg.value_normalizer_decay,
                                       disable=not cfg.normalize_values)
        value_normalizer = value_normalizer.to(dev)
        
        # Create rollout manager for this policy
        rollout_mgr = RolloutManager(dev, sim, cfg.steps_per_update,
            cfg.num_bptt_chunks, amp, actor_critic.recurrent_cfg)
        
        policies.append(policy)
        optimizers.append(optimizer)
        value_normalizers.append(value_normalizer)
        rollout_managers.append(rollout_mgr)

    parallel_state = ParallelPolicyState(
        policies=policies,
        optimizers=optimizers,
        value_normalizers=value_normalizers,
        rollout_managers=rollout_managers
    )

    if restore_ckpt != None:
        # Load checkpoint into all policies
        for policy in policies:
            policy.load_state_dict(torch.load(restore_ckpt))
        start_update_idx = 0
    else:
        start_update_idx = 0

    rollout_mgr = RolloutManager(dev, sim, cfg.steps_per_update,
        cfg.num_bptt_chunks, amp, actor_critic.recurrent_cfg)

    if dev.type == 'cuda':
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

def train_parallel(dev, sim, cfg, actor_critic, update_cb, restore_ckpt=None, num_parallel_policies=4):
    """Train multiple policies in parallel and select the best one at each step."""
    print(f"Starting parallel training with {num_parallel_policies} policies")
    print(cfg)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    num_agents = sim.actions.shape[0]
    total_worlds = sim.actions.shape[0] // 2  # Assuming 2 agents per world
    worlds_per_policy = total_worlds // num_parallel_policies

    # Create multiple policies with different hyperparameters
    policies = []
    optimizers = []
    value_normalizers = []
    rollout_managers = []
    
    # Default hyperparameters
    default_lr = cfg.lr
    default_gamma = cfg.gamma
    default_entropy = cfg.ppo.entropy_coef
    default_value = cfg.ppo.value_loss_coef
    
    # Initialize amp before creating rollout managers
    amp = AMPState(dev, cfg.mixed_precision)
    
    for i in range(num_parallel_policies):
        # Generate random hyperparameters using log-normal distribution
        scale = 0.5
        
        # Generate random values in log space
        policy_lr = default_lr * math.exp(torch.randn(1).item() * scale)
        policy_gamma = default_gamma * math.exp(torch.randn(1).item() * scale)
        policy_entropy = default_entropy * math.exp(torch.randn(1).item() * scale)
        policy_value = default_value * math.exp(torch.randn(1).item() * scale)

        if num_parallel_policies >= 1:
            policy_lr = cfg.lr
            policy_gamma = cfg.gamma
            policy_entropy = cfg.ppo.entropy_coef
            policy_value = cfg.ppo.value_loss_coef
        
        # Clamp gamma to valid range
        policy_gamma = max(0.9, min(0.999, policy_gamma))
        
        # Create policy with modified config
        policy = actor_critic.to(dev)
        optimizer = optim.Adam(policy.parameters(), lr=policy_lr)
        value_normalizer = EMANormalizer(cfg.value_normalizer_decay,
                                       disable=not cfg.normalize_values)
        value_normalizer = value_normalizer.to(dev)
        
        # Create rollout manager for this policy with correct number of worlds
        rollout_mgr = RolloutManager(dev, sim, cfg.steps_per_update,
            cfg.num_bptt_chunks, amp, actor_critic.recurrent_cfg)
        
        # Store hyperparameters in policy for logging
        policy.hyperparams = {
            'lr': policy_lr,
            'gamma': policy_gamma,
            'entropy_coef': policy_entropy,
            'value_loss_coef': policy_value,
            'worlds_per_policy': worlds_per_policy
        }
        
        policies.append(policy)
        optimizers.append(optimizer)
        value_normalizers.append(value_normalizer)
        rollout_managers.append(rollout_mgr)

    parallel_state = ParallelPolicyState(
        policies=policies,
        optimizers=optimizers,
        value_normalizers=value_normalizers,
        rollout_managers=rollout_managers
    )

    if restore_ckpt != None:
        # Load checkpoint into all policies
        for policy in policies:
            policy.load_state_dict(torch.load(restore_ckpt))
        start_update_idx = 0
    else:
        start_update_idx = 0

    if dev.type == 'cuda':
        def gpu_sync_fn():
            torch.cuda.synchronize()
    else:
        def gpu_sync_fn():
            pass

    def update_iter_wrapper(cfg, amp, num_train_seqs, sim, rollout_mgr, advantages, parallel_state, scheduler, update_idx, writer):
        return _update_iter(cfg, amp, num_train_seqs, sim, rollout_mgr, advantages, parallel_state, scheduler, update_idx, writer)

    _update_loop(
        update_iter_fn=update_iter_wrapper,
        gpu_sync_fn=gpu_sync_fn,
        user_cb=update_cb,
        cfg=cfg,
        num_agents=num_agents,
        sim=sim,
        rollout_mgr=rollout_managers[0],  # Use first rollout manager as default
        parallel_state=parallel_state,
        start_update_idx=start_update_idx,
        writer=None,
    )

    # Return best policy
    return parallel_state.policies[parallel_state.best_policy_idx]
