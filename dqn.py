import gym
import collections
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque()
        self.batch_size = 32
        self.size_limit = 50000
    
    def put(self, data):
        self.buffer.append(data)
        if len(self.buffer) > self.size_limit:
            self.buffer.popleft()
    
    def sample(self, n):
        return random.sample(self.buffer, n)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()
            
def train(q, q_target, memory, gamma, optimizer, batch_size):
    for i in range(10):
        batch = memory.sample(batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        s,a,r,s_prime,done_mask = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), torch.tensor(done_mask_lst)
        q_out = q(s)
        q_a = q_out.gather(1,a)
        q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * q_prime * done_mask
        loss = F.smooth_l1_loss(target, q_a)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    q_target.eval()
    memory = ReplayBuffer()

    avg_t = 0
    gamma = 0.98
    batch_size = 32
    optimizer = optim.Adam(q.parameters(), lr=0.0005)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = env.reset()
        for t in range(600):
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)      
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/200.0,s_prime, done_mask))
            s = s_prime
            if done:
                break

        avg_t += t
        if memory.size()>2000:
            train(q, q_target, memory, gamma, optimizer, batch_size)

        if n_epi%20==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("# of episode :{}, Avg timestep : {:.1f}, buffer size : {}, epsilon : {:.1f}%".format(n_epi, avg_t/20.0, memory.size(), epsilon*100))
            avg_t = 0
    env.close()

if __name__ == '__main__':
    main()