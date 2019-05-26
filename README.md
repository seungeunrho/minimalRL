# minimalRL-pytorch

Implementations of basic RL algorithms with minimal lines of codes! (PyTorch based)

* Each algorithm is complete within a single file.

* Every algorithm can be trained within 30 seconds, even without GPU.

* Envs are fixed to "CartPole-v1". You can just focus on the implementations.



## Algorithms
1. REINFORCE (66 lines)
2. TD Actor-Critic (97 lines)
3. DQN (113 lines,  including replay memory and target network)
4. PPO (116 lines,  including GAE)
5. DDPG (149 lines, including OU noise and soft target update)
6. A3C (116 lines)
7. Any suggestion..?


## Dependencies
1. PyTorch
2. OpenAI GYM

## Usage
```bash
# Works only with Python 3.
# e.g.
python3 REINFORCE.py
python3 actor_critic.py
python3 dqn.py
python3 ppo.py
python3 ddpg.py
python3 a3c.py
```
