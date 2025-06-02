import torch
import madrona_escape_room
import random
import copy
import numpy as np
from typing import List, Tuple
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

from madrona_escape_room_learn import (
    train, profile, TrainConfig, PPOConfig, PBTConfig, SimInterface,
)

from policy import make_policy, setup_obs

import argparse
import math
from pathlib import Path

torch.manual_seed(0)

class PopulationManager:
    def __init__(self, cfg: TrainConfig, num_obs_features: int, num_channels: int, separate_value: bool, writer: SummaryWriter):
        self.cfg = cfg
        self.population: List[Tuple[torch.nn.Module, torch.optim.Optimizer, float]] = []  # (policy, optimizer, fitness)
        self.best_fitness = float('-inf')
        self.best_policy = None
        self.writer = writer
        self.evolution_step = 0
        
        # Initialize population
        for i in range(cfg.pbt.population_size):
            policy = make_policy(num_obs_features, num_channels, separate_value)
            optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
            self.population.append((policy, optimizer, float('-inf')))
            # Log initial hyperparameters
            self.writer.add_scalar(f'population/policy_{i}/learning_rate', 
                                 optimizer.param_groups[0]['lr'],
                                 self.evolution_step)
    
    def evaluate_policy(self, policy_idx: int, sim: madrona_escape_room.SimManager, dev: torch.device) -> float:
        policy, optimizer, _ = self.population[policy_idx]
        policy = policy.to(dev)
        
        # Run evaluation episodes
        total_reward = 0
        num_episodes = 5
        
        for _ in range(num_episodes):
            # Reset simulation using reset tensor
            reset_tensor = sim.reset_tensor()
            reset_tensor.to_torch().fill_(1)
            sim.step()
            
            # Get initial observations exactly as in setup_obs
            self_obs_tensor = sim.self_observation_tensor().to_torch()
            partner_obs_tensor = sim.partner_observations_tensor().to_torch()
            room_ent_obs_tensor = sim.room_entity_observations_tensor().to_torch()
            door_obs_tensor = sim.door_observation_tensor().to_torch()
            lidar_tensor = sim.lidar_tensor().to_torch()
            steps_remaining_tensor = sim.steps_remaining_tensor().to_torch()

            N, A = self_obs_tensor.shape[0:2]
            batch_size = N * A

            # Create agent ID tensor exactly as in setup_obs
            id_tensor = torch.arange(A).float()
            if A > 1:
                id_tensor = id_tensor / (A - 1)
            id_tensor = id_tensor.to(device=dev)
            id_tensor = id_tensor.view(1, A).expand(N, A).reshape(batch_size, 1)

            # Format observations exactly as in setup_obs
            obs_tensors = [
                self_obs_tensor.view(batch_size, *self_obs_tensor.shape[2:]),
                partner_obs_tensor.view(batch_size, *partner_obs_tensor.shape[2:]),
                door_obs_tensor.view(batch_size, *door_obs_tensor.shape[2:]),
                room_ent_obs_tensor.view(batch_size, *room_ent_obs_tensor.shape[2:]),
                lidar_tensor.view(batch_size, *lidar_tensor.shape[2:]),
                steps_remaining_tensor.view(batch_size, *steps_remaining_tensor.shape[2:]),
                id_tensor,
            ]
            
            # Move all tensors to device
            obs_tensors = [obs.to(dev) for obs in obs_tensors]
            
            # Initialize RNN states
            rnn_states = None
            
            done = False
            episode_reward = 0
            
            while not done:
                with torch.no_grad():
                    # Get action from policy with RNN states
                    action_dists, _, rnn_states = policy(rnn_states, *obs_tensors)
                    # Get best action from distribution
                    action = torch.zeros(batch_size, len(action_dists.dists), dtype=torch.long, device=dev)
                    action_dists.best(action)
                    
                    # Set action in simulation tensor
                    sim_action = sim.action_tensor().to_torch()
                    sim_action.copy_(action.view(sim_action.shape))
                    
                # Step simulation
                sim.step()
                
                # Get new observations exactly as above
                self_obs_tensor = sim.self_observation_tensor().to_torch()
                partner_obs_tensor = sim.partner_observations_tensor().to_torch()
                room_ent_obs_tensor = sim.room_entity_observations_tensor().to_torch()
                door_obs_tensor = sim.door_observation_tensor().to_torch()
                lidar_tensor = sim.lidar_tensor().to_torch()
                steps_remaining_tensor = sim.steps_remaining_tensor().to_torch()

                # Format observations exactly as above
                obs_tensors = [
                    self_obs_tensor.view(batch_size, *self_obs_tensor.shape[2:]),
                    partner_obs_tensor.view(batch_size, *partner_obs_tensor.shape[2:]),
                    door_obs_tensor.view(batch_size, *door_obs_tensor.shape[2:]),
                    room_ent_obs_tensor.view(batch_size, *room_ent_obs_tensor.shape[2:]),
                    lidar_tensor.view(batch_size, *lidar_tensor.shape[2:]),
                    steps_remaining_tensor.view(batch_size, *steps_remaining_tensor.shape[2:]),
                    id_tensor,
                ]
                
                # Move all tensors to device
                obs_tensors = [obs.to(dev) for obs in obs_tensors]
                
                # Get reward
                reward = sim.reward_tensor().to_torch().mean().item()
                done = sim.done_tensor().to_torch().any().item()
                
                episode_reward += reward
            
            total_reward += episode_reward
        
        avg_reward = total_reward / num_episodes
        self.population[policy_idx] = (policy, optimizer, avg_reward)
        
        # Log policy performance
        self.writer.add_scalar(f'population/policy_{policy_idx}/fitness', avg_reward, self.evolution_step)
        
        if avg_reward > self.best_fitness:
            self.best_fitness = avg_reward
            self.best_policy = copy.deepcopy(policy)
            self.writer.add_scalar('population/best_fitness', self.best_fitness, self.evolution_step)
        
        return avg_reward
    
    def evolve_population(self):
        if not self.cfg.pbt:
            return
            
        # Sort by fitness
        self.population.sort(key=lambda x: x[2], reverse=True)
        
        # Keep elite policies
        num_elite = max(1, int(self.cfg.pbt.population_size * self.cfg.pbt.elite_fraction))
        elite = self.population[:num_elite]
        
        # Create new population
        new_population = []
        
        # Add elite policies
        new_population.extend(elite)
        
        # Fill rest with mutated copies of best policies
        while len(new_population) < self.cfg.pbt.population_size:
            parent_idx = random.randint(0, num_elite - 1)
            parent_policy, parent_optimizer, _ = self.population[parent_idx]
            
            # Create mutated copy
            child_policy = copy.deepcopy(parent_policy)
            child_optimizer = torch.optim.Adam(child_policy.parameters(), lr=parent_optimizer.param_groups[0]['lr'])
            
            # Mutate hyperparameters
            if random.random() < self.cfg.pbt.mutation_rate:
                # Mutate learning rate
                old_lr = child_optimizer.param_groups[0]['lr']
                new_lr = old_lr * random.uniform(
                    1 - self.cfg.pbt.mutation_scale,
                    1 + self.cfg.pbt.mutation_scale
                )
                child_optimizer.param_groups[0]['lr'] = new_lr
                
                # Log mutation
                self.writer.add_scalar(f'population/policy_{len(new_population)}/learning_rate', 
                                     new_lr, self.evolution_step)
            
            new_population.append((child_policy, child_optimizer, float('-inf')))
        
        self.population = new_population
        self.evolution_step += 1

class LearningCallback:
    def __init__(self, ckpt_dir, profile_report, population_manager: PopulationManager, writer: SummaryWriter, cfg: TrainConfig):
        self.mean_fps = 0
        self.ckpt_dir = ckpt_dir
        self.profile_report = profile_report
        self.population_manager = population_manager
        self.update_count = 0
        self.writer = writer
        self.cfg = cfg

    def __call__(self, update_idx, update_time, update_results, learning_state):
        update_id = update_idx + 1
        self.update_count += 1
        fps = args.num_worlds * args.steps_per_update / update_time
        self.mean_fps += (fps - self.mean_fps) / update_id

        if update_id != 1 and update_id % 10 != 0:
            return

        ppo = update_results.ppo_stats

        with torch.no_grad():
            reward_mean = update_results.rewards.mean().cpu().item()
            reward_min = update_results.rewards.min().cpu().item()
            reward_max = update_results.rewards.max().cpu().item()

            value_mean = update_results.values.mean().cpu().item()
            value_min = update_results.values.min().cpu().item()
            value_max = update_results.values.max().cpu().item()

            advantage_mean = update_results.advantages.mean().cpu().item()
            advantage_min = update_results.advantages.min().cpu().item()
            advantage_max = update_results.advantages.max().cpu().item()

            bootstrap_value_mean = update_results.bootstrap_values.mean().cpu().item()
            bootstrap_value_min = update_results.bootstrap_values.min().cpu().item()
            bootstrap_value_max = update_results.bootstrap_values.max().cpu().item()

            vnorm_mu = learning_state.value_normalizer.mu.cpu().item()
            vnorm_sigma = learning_state.value_normalizer.sigma.cpu().item()

        # Log metrics to TensorBoard
        self.writer.add_scalar('training/loss', ppo.loss, update_id)
        self.writer.add_scalar('training/action_loss', ppo.action_loss, update_id)
        self.writer.add_scalar('training/value_loss', ppo.value_loss, update_id)
        self.writer.add_scalar('training/entropy_loss', ppo.entropy_loss, update_id)
        
        self.writer.add_scalar('rewards/mean', reward_mean, update_id)
        self.writer.add_scalar('rewards/min', reward_min, update_id)
        self.writer.add_scalar('rewards/max', reward_max, update_id)
        
        self.writer.add_scalar('values/mean', value_mean, update_id)
        self.writer.add_scalar('values/min', value_min, update_id)
        self.writer.add_scalar('values/max', value_max, update_id)
        
        self.writer.add_scalar('advantages/mean', advantage_mean, update_id)
        self.writer.add_scalar('advantages/min', advantage_min, update_id)
        self.writer.add_scalar('advantages/max', advantage_max, update_id)
        
        self.writer.add_scalar('bootstrap_values/mean', bootstrap_value_mean, update_id)
        self.writer.add_scalar('bootstrap_values/min', bootstrap_value_min, update_id)
        self.writer.add_scalar('bootstrap_values/max', bootstrap_value_max, update_id)
        
        self.writer.add_scalar('returns/mean', ppo.returns_mean, update_id)
        self.writer.add_scalar('returns/stddev', ppo.returns_stddev, update_id)
        
        self.writer.add_scalar('value_normalizer/mean', vnorm_mu, update_id)
        self.writer.add_scalar('value_normalizer/stddev', vnorm_sigma, update_id)
        
        self.writer.add_scalar('performance/fps', fps, update_id)
        self.writer.add_scalar('performance/mean_fps', self.mean_fps, update_id)
        self.writer.add_scalar('performance/update_time', update_time, update_id)

        print(f"\nUpdate: {update_id}")
        print(f"    Loss: {ppo.loss: .3e}, A: {ppo.action_loss: .3e}, V: {ppo.value_loss: .3e}, E: {ppo.entropy_loss: .3e}")
        print()
        print(f"    Rewards          => Avg: {reward_mean: .3e}, Min: {reward_min: .3e}, Max: {reward_max: .3e}")
        print(f"    Values           => Avg: {value_mean: .3e}, Min: {value_min: .3e}, Max: {value_max: .3e}")
        print(f"    Advantages       => Avg: {advantage_mean: .3e}, Min: {advantage_min: .3e}, Max: {advantage_max: .3e}")
        print(f"    Bootstrap Values => Avg: {bootstrap_value_mean: .3e}, Min: {bootstrap_value_min: .3e}, Max: {bootstrap_value_max: .3e}")
        print(f"    Returns          => Avg: {ppo.returns_mean}, σ: {ppo.returns_stddev}")
        print(f"    Value Normalizer => Mean: {vnorm_mu: .3e}, σ: {vnorm_sigma :.3e}")

        if self.profile_report:
            print()
            print(f"    FPS: {fps:.0f}, Update Time: {update_time:.2f}, Avg FPS: {self.mean_fps:.0f}")
            print(f"    PyTorch Memory Usage: {torch.cuda.memory_reserved() / 1024 / 1024 / 1024:.3f}GB (Reserved), {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.3f}GB (Current)")
            profile.report()

        if update_id % 100 == 0:
            learning_state.save(update_idx, self.ckpt_dir / f"{update_id}.pth")
            
        # PBT evolution
        if self.population_manager and self.cfg.pbt and self.update_count % self.cfg.pbt.eval_interval == 0:
            print("\nEvaluating population...")
            for i in range(len(self.population_manager.population)):
                fitness = self.population_manager.evaluate_policy(i, sim, dev)
                print(f"Policy {i} fitness: {fitness:.2f}")
            
            print("\nEvolving population...")
            self.population_manager.evolve_population()
            print(f"Best fitness so far: {self.population_manager.best_fitness:.2f}")

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--ckpt-dir', type=str, required=True)
arg_parser.add_argument('--restore', type=int)

arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-updates', type=int, required=True)
arg_parser.add_argument('--steps-per-update', type=int, default=40)
arg_parser.add_argument('--num-bptt-chunks', type=int, default=8)

arg_parser.add_argument('--lr', type=float, default=1e-4)
arg_parser.add_argument('--gamma', type=float, default=0.998)
arg_parser.add_argument('--entropy-loss-coef', type=float, default=0.01)
arg_parser.add_argument('--value-loss-coef', type=float, default=0.5)
arg_parser.add_argument('--clip-value-loss', action='store_true')

arg_parser.add_argument('--num-channels', type=int, default=256)
arg_parser.add_argument('--separate-value', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')

arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--profile-report', action='store_true')

# PBT arguments
arg_parser.add_argument('--pbt', action='store_true', help='Enable population based training')
arg_parser.add_argument('--population-size', type=int, default=8, help='Number of policies in population')
arg_parser.add_argument('--elite-fraction', type=float, default=0.2, help='Fraction of elite policies to keep')
arg_parser.add_argument('--mutation-rate', type=float, default=0.1, help='Probability of mutating hyperparameters')
arg_parser.add_argument('--mutation-scale', type=float, default=0.2, help='Scale of hyperparameter mutations')
arg_parser.add_argument('--eval-interval', type=int, default=500, help='How often to evaluate and evolve population')

# TensorBoard arguments
arg_parser.add_argument('--experiment-name', type=str, default=None, help='Name for this experiment run')
arg_parser.add_argument('--log-dir', type=str, default='runs', help='Directory to store TensorBoard logs')

args = arg_parser.parse_args()

# Setup TensorBoard
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Generate experiment name based on key hyperparameters
name_parts = []
name_parts.append("escape_room")

# Add PBT info if enabled
if args.pbt:
    name_parts.append(f"pbt{args.population_size}")
    name_parts.append(f"elite{int(args.elite_fraction*100)}")
    name_parts.append(f"mut{int(args.mutation_rate*100)}")

# Add key training parameters
name_parts.append(f"lr{args.lr:.0e}")
name_parts.append(f"gamma{args.gamma:.3f}")
name_parts.append(f"ent{args.entropy_loss_coef:.2f}")
name_parts.append(f"val{args.value_loss_coef:.2f}")

# Add model architecture
name_parts.append(f"ch{args.num_channels}")
if args.separate_value:
    name_parts.append("sep")

# Add training setup
name_parts.append(f"w{args.num_worlds}")
name_parts.append(f"u{args.num_updates}")
name_parts.append(f"s{args.steps_per_update}")

# Add precision info
if args.fp16:
    name_parts.append("fp16")

# Add timestamp
name_parts.append(timestamp)

# Combine all parts
experiment_name = "_".join(name_parts)

# Override with user name if provided
if args.experiment_name:
    experiment_name = args.experiment_name

log_dir = os.path.join(args.log_dir, experiment_name)
writer = SummaryWriter(log_dir)

# Log hyperparameters
hparams = {k: v for k, v in vars(args).items() if not k.startswith('_')}
writer.add_hparams(hparams, {'dummy': 0})  # Dummy metric required by TensorBoard

sim = madrona_escape_room.SimManager(
    exec_mode = madrona_escape_room.madrona.ExecMode.CUDA if args.gpu_sim else madrona_escape_room.madrona.ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    rand_seed = 5,
    auto_reset = True,
)

ckpt_dir = Path(args.ckpt_dir)

obs, num_obs_features = setup_obs(sim)

# Initialize PBT if enabled
population_manager = None
if args.pbt:
    pbt_config = PBTConfig(
        population_size=args.population_size,
        elite_fraction=args.elite_fraction,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        eval_interval=args.eval_interval
    )
    population_manager = PopulationManager(
        TrainConfig(
            num_updates=args.num_updates,
            steps_per_update=args.steps_per_update,
            num_bptt_chunks=args.num_bptt_chunks,
            lr=args.lr,
            gamma=args.gamma,
            ppo=PPOConfig(
                num_mini_batches=1,
                clip_coef=0.2,
                value_loss_coef=args.value_loss_coef,
                entropy_coef=args.entropy_loss_coef,
                max_grad_norm=0.5,
                num_epochs=2,
                clip_value_loss=args.clip_value_loss,
            ),
            pbt=pbt_config,
            value_normalizer_decay=0.999,
            mixed_precision=args.fp16,
        ),
        num_obs_features,
        args.num_channels,
        args.separate_value,
        writer
    )
    policy = population_manager.population[0][0]  # Start with first policy
else:
    policy = make_policy(num_obs_features, args.num_channels, args.separate_value)

train_config = TrainConfig(
    num_updates=args.num_updates,
    steps_per_update=args.steps_per_update,
    num_bptt_chunks=args.num_bptt_chunks,
    lr=args.lr,
    gamma=args.gamma,
    gae_lambda=0.95,
    ppo=PPOConfig(
        num_mini_batches=1,
        clip_coef=0.2,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_loss_coef,
        max_grad_norm=0.5,
        num_epochs=2,
        clip_value_loss=args.clip_value_loss,
    ),
    pbt=pbt_config if args.pbt else None,
    value_normalizer_decay=0.999,
    mixed_precision=args.fp16,
)

learning_cb = LearningCallback(ckpt_dir, args.profile_report, population_manager, writer, train_config)

if torch.cuda.is_available():
    dev = torch.device(f'cuda:{args.gpu_id}')
else:
    dev = torch.device('cpu')

ckpt_dir.mkdir(exist_ok=True, parents=True)

actions = sim.action_tensor().to_torch()
dones = sim.done_tensor().to_torch()
rewards = sim.reward_tensor().to_torch()

# Flatten N, A, ... tensors to N * A, ...
actions = actions.view(-1, *actions.shape[2:])
dones  = dones.view(-1, *dones.shape[2:])
rewards = rewards.view(-1, *rewards.shape[2:])

if args.restore:
    restore_ckpt = ckpt_dir / f"{args.restore}.pth"
else:
    restore_ckpt = None

train(
    dev,
    SimInterface(
            step = lambda: sim.step(),
            obs = obs,
            actions = actions,
            dones = dones,
            rewards = rewards,
    ),
    train_config,
    policy,
    learning_cb,
    restore_ckpt
)

writer.close()
