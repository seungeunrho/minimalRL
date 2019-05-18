# minimalRL-pytorch

Implementations of basic RL algorithms with minimal lines of codes! (PyTorch based)

Every environment is fixed to "CartPole-v1" from OpenAI GYM. You can just focus on the implementations.


2019-05-17 : PPO added (116 lines,  including GAE)

2019-05-06 : DQN added (113 lines,  including replay memory and target network)

2019-04-27 : TD Actor-Critic added (74 lines)

2019-04-23 : REINFORCE added (64 lines)





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
```
