# minimalRL-pytorch

Implementations of basic RL algorithms with minimal lines of codes! (PyTorch based)

* Each algorithm is complete within a single file.

* Length of each file is up to 100~150 lines of codes.

* Every algorithm can be trained within 30 seconds, even without GPU.

* Envs are fixed to "CartPole-v1". You can just focus on the implementations.



## Algorithms
1. [REINFORCE](https://github.com/seungeunrho/minimalRL/blob/master/REINFORCE.py) (67 lines)
2. [Vanilla Actor-Critic](https://github.com/seungeunrho/minimalRL/blob/master/actor_critic.py) (98 lines)
3. [DQN](https://github.com/seungeunrho/minimalRL/blob/master/dqn.py) (112 lines,  including replay memory and target network)
4. [PPO](https://github.com/seungeunrho/minimalRL/blob/master/ppo.py) (119 lines,  including GAE)
5. [DDPG](https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py) (145 lines, including OU noise and soft target update)
6. [A3C](https://github.com/seungeunrho/minimalRL/blob/master/a3c.py) (129 lines)
7. [ACER](https://github.com/seungeunrho/minimalRL/blob/master/acer.py) (149 lines)
8. [A2C](https://github.com/seungeunrho/minimalRL/blob/master/a2c.py) (188 lines)
9. [SAC](https://github.com/seungeunrho/minimalRL/blob/master/sac.py) (171 lines) added!! 
10. [PPO-Continuous](https://github.com/seungeunrho/minimalRL/blob/master/ppo-continuous.py) (161 lines) added!!
11. Any suggestion ...? 


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
python3 a2c.py
python3 acer.py
python3 sac.py
```
