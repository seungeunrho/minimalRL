#Vanilla TD Actor-Critic
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.loss_lst = []
        
        self.fc1 = nn.Linear(4, 128)
        self.fc_pi = nn.Linear(128, 2)
        self.fc_v = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.005)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        pol = self.fc_pi(x)
        pi = F.softmax(pol, dim=0)
        v = self.fc_v(x)
        return pi, v
    
    def gather_loss(self, loss):
        self.loss_lst.append(loss.unsqueeze(0))
    
    def train(self):
        loss = torch.cat(self.loss_lst).sum()
        loss = loss/len(self.loss_lst)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_lst = []
        

def main():
    env = gym.make('CartPole-v1')
    model = ActorCritic()
    gamma = 0.99

    score = 0.0
    print_interval = 30

    for n_epi in range(10000):
        obs = env.reset()
        loss_lst = []
        for t in range(600):
            obs = torch.from_numpy(obs).float()
            pi, v = model(obs)
            m = Categorical(pi)
            action = m.sample()

            obs, r, done, info = env.step(action.item())
            _, next_v = model(torch.from_numpy(obs).float())
            delta = r + gamma * next_v - v
            loss = -torch.log(pi[action]) * delta.item() + delta * delta.item()
            model.gather_loss(loss)

            score += r
            if done:
                break
        
        model.train()
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()
