import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20
mini_batch_size = 64
class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
    def choose_mini_batch(self, mini_batch_size,s_, a_, r_, s_prime_, done_mask_, log_prob_,advantage_,returns):
        full_batch_size = len(s_)
        for _ in range(full_batch_size // mini_batch_size):
            indices = np.random.randint(0, full_batch_size, mini_batch_size)
            yield s_[indices], a_[indices], r_[indices], s_prime_[indices], done_mask_[indices],\
                  log_prob_[indices],advantage_[indices], returns[indices]
    def train_net(self):
        s_, a_, r_, s_prime_, done_mask_, old_log_prob_ = self.make_batch()
        td_target = r_ + gamma * self.v(s_prime_) * done_mask_
        delta = td_target - self.v(s_)
        delta = delta.detach().numpy()
        advantage_lst = []
        advantage = 0.0
        for idx in reversed(range(len(delta))):
            advantage = gamma * lmbda * advantage * done_mask_[idx] + delta[idx][0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage_ = torch.tensor(advantage_lst, dtype=torch.float)
        returns = advantage_ + self.v(s_)
        advantage_ = (advantage_ - advantage_.mean())/(advantage_.std()+1e-3)
        for i in range(K_epoch):
            for s,a,r,s_prime,done_mask,prob_a,advantage,return_ in self.choose_mini_batch(mini_batch_size,s_, a_, r_, s_prime_, done_mask_, old_log_prob_,advantage_,returns): 
                pi = self.pi(s, softmax_dim=1)
                pi_a = pi.gather(1,a)
                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , return_.detach())

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
        
def main():
    env = gym.make('CartPole-v1')
    model = PPO()
    score = 0.0
    print_interval = 20
    T_horizon = 1024
    score_lst = []
    for n_epi in range(10000):
        score = 0.0
        s = env.reset()
        done = False
        for t in range(T_horizon):
            prob = model.pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample().item()
            s_prime, r, done, info = env.step(a)

            model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
            s = s_prime

            score += r
            if done:
                s = (env.reset())
                score_lst.append(score)
                score = 0
            else:
                s = s_prime

        model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
            score_lst = []
    env.close()

if __name__ == '__main__':
    main()
