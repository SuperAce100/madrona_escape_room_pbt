Madrona Escape Room + PBT
============================

The Environment and Learning Task
--------------

https://github.com/shacklettbp/madrona_escape_room/assets/1111429/ec6231c8-a74b-4f0a-8a1a-b1bcdc7111cd

As shown above, the simulator implements a 3D environment consisting of two agents and a row of three rooms. All agents start in the first room, and must navigate to as many new rooms as possible. The agents must step on buttons or push movable blocks over buttons to trigger the opening of doors that lead to new rooms. Agents are rewarded based on their progress along the length of the level.

The codebase trains a shared policy that controls agents individually with direct engine inputs rather than pixel observations. Agents interact with the simulator as follows:

**Action Space:**
 * Movement amount: Egocentric polar coordinates for the direction and amount to move, translated to XY forces in the physics engine.
 * Rotation amount: Torque applied to the agent to turn.
 * Grab: Boolean, true to grab if possible or release if already holding an object.

**Observation Space:**
 * Global position.
 * Position within the current room.
 * Distance and direction to all the buttons and cubes in the current room (egocentric polar coordinates).
 * 30 Lidar samples arrayed in a circle around the agent, giving distance to the nearest object along a direction.
 * Whether the current room's door is open (boolean).
 * Whether an object is currently grabbed (boolean).
 * The max distance achieved so far in the level.
 * The number of steps remaining in the episode.

**Rewards:**
  Agents are rewarded for the max distance achieved along the Y axis (the length of the level). Each step, new reward is assigned if the agents have progressed further in the level, or a small penalty reward is assigned if not.
 
For specific details about the format of observations, refer to exported ECS components introduced in the [code walkthrough section](#simulator-code-walkthrough-learning-the-madrona-ecs-apis). 

Overall the "full simulator" contains logic for three major concerns:
* Procedurally generating a new random level for each episode.
* Time stepping the environment, which includes executing rigid body physics and evaluating game logic in response to agent actions.
* Generating agent observations from the state of the environment, which are communicated as PyTorch tensors to external policy evaluation or learning code.

Population Based Training (PBT)
--------------------------------

The main implementation of PBT can be found in [train_src/madrona_escape_room_learn/train.py] and [scripts/train.py].

This implementation uses Population Based Training (PBT) to optimize policy hyperparameters through evolutionary search. The algorithm maintains a population of N policies, each with independent hyperparameters, and periodically evolves them based on performance.

### Algorithm

1. **Fitness Evaluation**
   - Each policy is evaluated over K=5 episodes
   - Fitness score = mean episode reward
   - Episodes run in parallel using the Madrona simulator
   - RNN states are properly managed across episodes
   - Evaluation uses deterministic action selection (best action from policy distribution)

2. **Evolution**
   - Sort policies by fitness
   - Preserve top M = ⌈N * elite_fraction⌉ policies
   - For remaining N-M policies:
     - Select parent from top M policies uniformly
     - Copy parent's weights
     - With probability mutation_rate:
       - Mutate learning rate: lr_new = lr_old * U(1 ± mutation_scale)
     - Initialize new optimizer with mutated hyperparameters

3. **Training Integration**
   - PPO updates continue between evolution steps
   - Evolution occurs every eval_interval updates
   - Best policy is checkpointed when fitness improves

### Usage

```bash
python scripts/train.py --pbt --population-size 8 --elite-fraction 0.2 \
    --mutation-rate 0.1 --mutation-scale 0.2 --eval-interval 500
```

The implementation is integrated with TensorBoard for monitoring population statistics, individual policy performance, and hyperparameter evolution.
