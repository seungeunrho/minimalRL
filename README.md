# minimalRL-pytorch

Implementations of basic RL algorithms with minimal lines of codes! (PyTorch based)

* Each algorithm is complete within a single file.

* Every algorithm can be trained within 30 seconds, even without GPU.

* Envs are fixed to "CartPole-v1". You can just focus on the implementations.



## Algorithms

1. 2019-05-23 : A3C added  (116 lines)
2. 2019-05-21 : DDPG added (149 lines, including OU noise and soft target update)
3. 2019-05-17 : PPO added  (116 lines,  including GAE)
4. 2019-05-06 : DQN added  (115 lines,  including replay memory and target network)
5. 2019-04-27 : TD Actor-Critic added (77 lines)
6. 2019-04-23 : REINFORCE added (67 lines)





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
