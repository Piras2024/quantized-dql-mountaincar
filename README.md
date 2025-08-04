# Quantized DQL on MountainCarContinuous-v0 🏔️🚗

This project implements Deep Q-Learning (DQL) on the `MountainCarContinuous-v0` environment from [Gymnasium](https://gymnasium.farama.org/). Since DQL is inherently suited for discrete action spaces, the continuous action space of the environment is quantized to varying resolutions.

## 📌 Objective

To evaluate the performance of Deep Q-Learning under different levels of action space quantization:
- 3 discrete actions
- 11 discrete actions
- 101 discrete actions
- 1001 discrete actions

## 🧠 Method

1. **Environment**: `MountainCarContinuous-v0`
2. **Algorithm**: Deep Q-Learning
3. **Action Space Quantization**:
   - Continuous action range [-1.0, 1.0] is split into `N` evenly spaced actions.
   - Actions are discretized using `np.linspace(-1.0, 1.0, N)`.
4. **Neural Network**: Simple feedforward Q-network
5. **Exploration**: ε-greedy strategy

## 📊 Goals

- Understand how quantization granularity affects:
  - Convergence speed
  - Maximum reward
  - Training stability
- Determine if there's a tradeoff between performance and action resolution

## 🏁 Results

Key observations (to be expanded after experiments):
- Coarse quantization (e.g., 3 actions) may lead to faster convergence but poor policy performance
- Finer quantization (e.g., 1001 actions) increases precision but slows training

## 📦 Requirements

```bash
pip install -r requirements.txt
