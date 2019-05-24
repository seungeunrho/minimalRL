# minimalRL-pytorch

Implementations of basic RL algorithms with minimal lines of codes! (PyTorch based)

* Each algorithm is complete within a single file.

* Every algorithm can be trained within 30 seconds, even without GPU.

* Envs are fixed to "CartPole-v1". You can just focus on the implementations.



## Algorithms
1. REINFORCE added (66 lines, 2019.04)
2. TD Actor-Critic added (97 lines, 2019.04)
3. DQN added  (113 lines,  including replay memory and target network, 2019.05)
4. PPO added  (116 lines,  including GAE, 2019.05)
5. DDPG added (149 lines, including OU noise and soft target update, 2019.05)
6. A3C added  (116 lines, 2019.05)


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
