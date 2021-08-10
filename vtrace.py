import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

#Hyperparameters
learning_rate      = 0.0005
gamma              = 0.98
T_horizon          = 20
clip_rho_threshold = 1.0
clip_c_threshold   = 1.0
print_interval     = 20

class Vtrace(nn.Module):
    def __init__(self):
        super(Vtrace, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.clip_rho_threshold = torch.tensor(clip_rho_threshold, dtype=torch.float)
        self.clip_c_threshold = torch.tensor(clip_c_threshold, dtype=torch.float)

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
        s_lst, a_lst, r_lst, s_prime_lst, mu_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, mu_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            mu_a_lst.append([mu_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, mu_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                        torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                        torch.tensor(done_lst, dtype=torch.float), torch.tensor(mu_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, mu_a

    def vtrace(self, s, a, r, s_prime, done_mask, mu_a):
        with torch.no_grad():
            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            v, v_prime = self.v(s), self.v(s_prime)
            ratio = torch.exp(torch.log(pi_a) - torch.log(mu_a))  # a/b == exp(log(a)-log(b))
            
            rhos = torch.min(self.clip_rho_threshold, ratio)
            cs = torch.min(self.clip_c_threshold, ratio).numpy()
            td_target = r + gamma * v_prime * done_mask
            delta = rhos*(td_target - v).numpy()
            
            vs_minus_v_xs_lst = []
            vs_minus_v_xs = 0.0
            vs_minus_v_xs_lst.append([vs_minus_v_xs])
            
            for i in range(len(delta)-1, -1, -1):
                vs_minus_v_xs = gamma * cs[i][0] * vs_minus_v_xs + delta[i][0]
                vs_minus_v_xs_lst.append([vs_minus_v_xs])
            vs_minus_v_xs_lst.reverse()
            
            vs_minus_v_xs = torch.tensor(vs_minus_v_xs_lst, dtype=torch.float)
            vs = vs_minus_v_xs[:-1] + v.numpy()
            vs_prime = vs_minus_v_xs[1:] + v_prime.numpy()
            advantage = r + gamma * vs_prime - v.numpy()
            
        return vs, advantage, rhos

    def train_net(self):
        s, a, r, s_prime, done_mask, mu_a = self.make_batch()
        vs, advantage, rhos = self.vtrace(s, a, r, s_prime, done_mask, mu_a)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
       
        val_loss = F.smooth_l1_loss(self.v(s) , vs)
        pi_loss = -rhos * torch.log(pi_a) * advantage
        loss =  pi_loss + val_loss

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        
def main():
    env = gym.make('CartPole-v1')
    model = Vtrace()
    score = 0.0
    
    for n_epi in range(10000):
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)

                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()