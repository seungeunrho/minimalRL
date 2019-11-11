import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import collections
import random
import numpy as np
import copy

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst,dtype = torch.float), \
               torch.tensor(r_lst,dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst,dtype=torch.float)
    
    def size(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self,min_log_std= -20, max_log_std = 2):
        super(Actor,self).__init__()
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.layer_1 = nn.Linear(3,64)
        self.layer_2 = nn.Linear(64,64)
        self.mu = nn.Linear(64,1)
        self.sigma = nn.Linear(64,1)
    def forward(self,x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        mu = self.mu(x)
        sigma = torch.exp(torch.clamp(self.sigma(x), self.min_log_std, self.max_log_std))
        dist = torch.distributions.normal.Normal(mu,sigma)
        return dist, mu, sigma
class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()
        self.critic_1 = nn.Linear(3,64)
        self.critic_2 = nn.Linear(64,64)
        self.critic_3 = nn.Linear(64,1)
    def forward(self,x):
        x = F.relu(self.critic_1(x))
        x = F.relu(self.critic_2(x))
        x = (self.critic_3(x))
        return x
class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, s, a):
        x = torch.cat((s, a), -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SAC(nn.Module):
    def __init__(self):
        super(SAC,self).__init__()
        
        self.actor = Actor()
        self.critic = Critic()
        self.target_critic = copy.copy(self.critic)
        self.q_1 = Q()
        self.q_2 = Q()
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = critic_lr)
        self.q_optimizer = optim.Adam(list(self.q_1.parameters()) + list(self.q_2.parameters()),lr = q_lr)
        
    def pi(self,x):
        action = self.actor(x)[0].sample()
        action = torch.tanh(action)
        return action
    def train_net(self):
        s,a,r,s_prime,done_mask  = memory.sample(batch_size)
        
        dist,mu,sigma = self.actor(s)
        sample = dist.sample()
        action = torch.tanh(sample)
        log_prob = dist.log_prob(sample) - torch.log(1- action.pow(2) + epsilon) 

        
        q_theta = torch.min(self.q_1(s,action), self.q_2(s,action))
        
        loss_v = ((self.critic(s) - (q_theta - log_prob).detach()) ** 2).mean()
        self.critic_optimizer.zero_grad()
        loss_v.backward()
        self.critic_optimizer.step()
        
        loss_q_1 = (self.q_1(s,a) - (r + done_mask * lambd * self.target_critic(s_prime).detach())) ** 2
        loss_q_2 = (self.q_2(s,a) - (r + done_mask * lambd * self.target_critic(s_prime).detach())) ** 2
        
        loss_q = (loss_q_1 + loss_q_2).mean()
        
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        loss_pi = (log_prob - q_theta).mean()
        
        self.actor_optimizer.zero_grad()
        loss_pi.backward()
        self.actor_optimizer.step()
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
actor_lr = 3e-4
critic_lr = 3e-4
q_lr = 3e-4
epsilon = torch.tensor(1e-7).float()
lambd = 0.99
batch_size = 128
tau = 0.005
buffer_limit = int(1e+6)
memory = ReplayBuffer()

env = gym.make('Pendulum-v0')
model = SAC()
memory = ReplayBuffer()

print_interval = 20

def main(render = False):
    score = 0.0

    for n_epi in range(10000):
        s = env.reset()
        global_step = 0
        done = False
        r = 0
        while not done:
            global_step += 1 
            if render:
                env.render()
            a = model.pi(torch.from_numpy(s).float()).item()
            s_prime, r, done, info = env.step([a* 2])
            
            memory.put((s,a,r/100.0,s_prime,False)) #done
            s = s_prime
            print('global_step : ',global_step, ' action : ',a,' reward : ',r,' done : ',done)
            score += r
            if done:
                break
        if memory.size()>2000:
            model.train_net()
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
main()