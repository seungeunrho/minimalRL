import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x,dim = 0):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=dim)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        s = torch.tensor(list(map(lambda x : x[0],self.data)),dtype= torch.float)
        a = torch.tensor(list(map(lambda x : [x[1]],self.data)))
        R = 0
<<<<<<< HEAD
        R_lst = []
        for _,_,r in self.data[::-1]:
            R = r + gamma * R
            R_lst.append(R)
        R_lst.reverse()
        R = torch.tensor(R_lst, dtype=torch.float).reshape(-1,1)

        prob = self.forward(s,dim = 1)
        prob = prob.gather(1,a)
        loss = - torch.log(prob) * R
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
=======
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
>>>>>>> upstream/master
        self.data = []

def main():
    env = gym.make('CartPole-v1')
    pi = Policy()
    score = 0.0
    print_interval = 100
    
    for n_epi in range(10000):
        s = env.reset()
        for t in range(501): # CartPole-v1 forced to terminates at 500 step.
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, info = env.step(a.item())
<<<<<<< HEAD
            pi.put_data((s,a,r))
=======
            pi.put_data((r,prob[a]))
            
>>>>>>> upstream/master
            s = s_prime
            score += r
            if done:
                break
        
        pi.train_net()
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()
    
if __name__ == '__main__':
    main()