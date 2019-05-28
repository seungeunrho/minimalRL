# minimalRL-pytorch

Implementations of basic RL algorithms with minimal lines of codes! (PyTorch based)

* Each algorithm is complete within a single file.

* Length of each file is up to 100~150 lines of codes.

* Every algorithm can be trained within 30 seconds, even without GPU.

* Envs are fixed to "CartPole-v1". You can just focus on the implementations.



## Algorithms
1. REINFORCE (67 lines)
2. Vanilla Actor-Critic (98 lines)
3. DQN (111 lines,  including replay memory and target network)
4. PPO (119 lines,  including GAE)
5. DDPG (147 lines, including OU noise and soft target update)
6. A3C (116 lines)
7. ACER added ! (149 lines)
8. Any suggestion..?


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
python3 acer.py
```
