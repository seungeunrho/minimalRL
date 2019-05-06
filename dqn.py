import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque()
        self.size_limit = 10000
    
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
        self.batch_size = 16
        self.gamma = 0.99
        
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)

    def forward(self, x):
        x = torch.from_numpy(x).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train(self, buffer):
        if buffer.size()>1000:
            for i in range(4):
                loss_lst = []
                sample = buffer.sample(self.batch_size)
                for item in sample:
                    s, a, r, s_prime, done = item
                    q = self.forward(s)
                    if done :
                        loss = F.smooth_l1_loss(0, q[a])
                    else:
                        target = r + self.gamma* self.forward(s_prime).max()
                        loss = F.smooth_l1_loss(target.item(), q[a])
                    loss_lst.append(loss.unsqueeze(0))
                loss = torch.cat(loss_lst).sum()
                loss = loss/len(loss_lst)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

def main():
    env = gym.make('CartPole-v1')
    q = Qnet()
    avg_t = 0
    memory = ReplayBuffer()

    for n_epi in range(10000):
        epsilon = max(0.01, 0.1 - 0.01*(n_epi/200)) #Linear annealing
        s = env.reset()
        for t in range(600):
            out = q(s)
            coin = random.random()
            if coin < epsilon:
                a = random.randint(0,1)
            else : 
                a = out.argmax().item()
            s_prime, r, done, info = env.step(a)
            memory.put((s,a,r/200.0,s_prime, done))
            s = s_prime
            
            if done:
                break
                
        avg_t += t
        q.train(memory)

        if n_epi%50==0 and n_epi!=0:
            print("# of episode :{}, Avg timestep : {}, buffer size : {}, epsilon : {:.1f}%".format(n_epi, avg_t/50.0, memory.size(), epsilon*100))
            avg_t = 0
    env.close()

if __name__ == '__main__':
    main()