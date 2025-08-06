# Quantized DQL on MountainCarContinuous-v0 🏔️🚗

This project implements Deep Q-Learning (DQL) on the `MountainCarContinuous-v0` environment from [Gymnasium](https://gymnasium.farama.org/). Since DQL is inherently suited for discrete action spaces, the continuous action space of the environment is quantized to varying resolutions.

## 🧠 Method

1. **Environment**  
   - `MountainCarContinuous-v0` from [Gymnasium](https://gymnasium.farama.org/)
   - The agent must learn to accelerate a car up a steep hill using limited engine power.

2. **Algorithm**  
   - **Deep Q-Learning (DQL)**: Applied to a continuous control problem by discretizing the action space.

3. **Action Space Quantization**  
   - The environment's continuous action range `[-1.0, 1.0]` is quantized into `N` evenly spaced actions.
   - Experiments were conducted with the following values for `N`: **3**, **11**, **101**, and **1001**.

4. **Training Setup**  
   - Each configuration was trained for **10,000 episodes**.
   - A **replay buffer** and **target network** were used to stabilize learning.

5. **Reward Shaping**  
   - To encourage faster episode termination, a reward of **-1 per timestep** was added.
   - For the **1001-action** case, an additional configuration was tested where:
     - A **positive reward was added every time the car reached a new maximum position on the right**.
     - This strategy **yielded poor results**, likely due to increased reward variance and ineffective guidance.

6. **Neural Network**  
   - A simple feedforward Q-network (Multi-Layer Perceptron) with:
     - Input: state features (position, velocity)
     - Output: Q-values for each quantized action
     - Hidden layers: standard MLP with ReLU activations
     - Optimizer: **Adam**

7. **Exploration Strategy**  
   - **ε-greedy exploration** was used to balance exploration and exploitation.
   - The default decay strategy was **hyperbolic**, using the formula:
     ```python
     epsilon = c1 / (c2 + i)
     ```
     where `i` is the episode number, and constants were set as `c1 = 1000`, `c2 = 1000`.
   - Other decay strategies were also tested:
     - **Linear decay** — yielded the best results for the 1001-actions case
     - **Exponential decay**
   - ε was typically annealed from `ε_start = 1.0` to `ε_min = 0.01`.

8. **Evaluation**  
   - After training, the best-performing policy **(found during training)** was evaluated over **20 test runs** to compute average reward.


## 📊 Results

![Training rewards](images/reward_per_episode.png)

As expected, increasing the action space granularity improved performance **up to a certain point**.  
The graph below (to be added) shows the **average reward of the best policy** found during training, evaluated over **20 test runs** for each quantization level.

![Average Test Reward Comparison](images/average_test_reward.png)

However, when the number of discrete actions becomes too high — such as **1001 actions** — training a successful agent using DQL becomes significantly harder.

Several strategies were explored to mitigate this issue:
- Different **ε-decay strategies**: linear, exponential, and hyperbolic
- **Learning rate schedules**
- Various **reward shaping techniques**

All experiment runs and configurations are available on my  
[Weights & Biases dashboard](https://wandb.ai/matteo-piras-universit-di-firenze/MountainCar%20DQL/workspace?nw=nwusermatteopiras).

The **best result** was obtained using:
- **10000 training episodes**
- **Linearly decaying ε**, which promotes more exploration

The core issue with very fine-grained (e.g., 1001) quantization appears to be **action dilution**:  
random action selection often picks values that **have little to no physical impact**, making it difficult for the agent to explore effectively and reach the reward zone.

Even when a reasonably good policy is found, the **average test reward remains low (≈ 55)**.  
This is because the policy is **unstable**: some episodes reach the goal with high reward, while others fail entirely.  
When averaged, these mixed outcomes result in a low overall reward — despite having high potential in some runs.
