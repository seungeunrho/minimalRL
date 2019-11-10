import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

class NoiseGenerator:
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma
    def generate(self):
        return np.random.normal(self.mu,self.sigma,1)[0]

class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))*2 
        return mu

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_3 = nn.Linear(32,1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_3(q)
        return q

    
class TD3(nn.Module):
    def __init__(self):
        super(TD3,self).__init__()
        self.q_net_1 = QNet()
        self.q_net_2 = QNet()
        self.q_net_optimizer = optim.Adam(list(self.q_net_1.parameters()) \
                                           + list(self.q_net_2.parameters()) ,lr = critic_learning_rate)
        self.mu_net = MuNet()
        self.mu_net_optimizer = optim.Adam(self.mu_net.parameters(), lr=actor_learning_rate)
        self.target_q_net_1 = copy.copy(self.q_net_1)
        self.target_q_net_2 = copy.copy(self.q_net_2)
        self.target_mu_net = copy.copy(self.mu_net)
        
        self.noise_generator = NoiseGenerator(0,0.2)
        
        self.train_num = 1
        self.d = 2
        
    def pi(self,x):
        x = self.mu_net(x)
        return x
    
    def train_net(self):
        s,a,r,s_prime,done_mask  = memory.sample(batch_size)
        tilde_a = self.target_mu_net(s_prime) + torch.tensor(np.clip(self.noise_generator.generate(),-0.5,0.5))
        tilde_a = torch.clamp(tilde_a,-2,2).detach()
        y = r + discount_factor * torch.min(self.target_q_net_1(s_prime,tilde_a),self.target_q_net_2(s_prime,tilde_a))
        q_loss = F.mse_loss(y.detach(), self.q_net_1(s,a)) + F.mse_loss(y.detach(), self.q_net_2(s,a))
        self.q_net_optimizer.zero_grad()
        q_loss.backward()
        self.q_net_optimizer.step()
        
        if self.train_num % self.d == 0 :
            mu_loss = - self.q_net_1(s,self.mu_net(s)).mean()
            self.mu_net_optimizer.zero_grad()
            mu_loss.backward()
            self.mu_net_optimizer.step()

            for param, target_param in zip(self.q_net_1.parameters(), self.target_q_net_1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.q_net_2.parameters(), self.target_q_net_2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.mu_net.parameters(), self.target_mu_net.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        self.train_num += 1

buffer_limit = 50000
actor_learning_rate = 0.001
critic_learning_rate = 0.001
batch_size = 100
discount_factor = 0.99
tau = 0.005

env = gym.make('Pendulum-v0')
model = TD3()
memory = ReplayBuffer()

print_interval = 20

def main(render = False):
    score = 0.0

    pi_noise_generator = NoiseGenerator(0,0.1)
    
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
            
            a = a + pi_noise_generator.generate()
            a = max(min(a, 2), -2)
            s_prime, r, done, info = env.step([a])
            memory.put((s,a,r/100.0,s_prime,done))
            s = s_prime
            #print('global_step : ',global_step, ' action : ',a,' reward : ',r,' done : ',done)
            score += r
            if done:
                break
        if memory.size()>2000:
            model.train_net()
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

main()