import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(3,64)
        self.fc_v  = nn.Linear(64,1)
        self.fc_pi = nn.Linear(64,1)
        self.fc_sigma = nn.Linear(64,1)
    
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x):
        x = F.tanh(self.fc1(x))
        mu = 2 * F.tanh(self.fc_pi(x))
        sigma = F.softplus(self.fc_sigma(x))

        normal_list = torch.distributions.normal.Normal(loc=mu,scale=sigma)
        return torch.clamp(normal_list.sample(),-2,2),mu,sigma
        
    def v(self, x):
        x = F.tanh(self.fc1(x))
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
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        global entropy
        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            _,new_mu,new_sigma = self.pi(s)
            
            pi_a_dist = torch.distributions.Normal(new_mu,new_sigma)
            pi_a = pi_a_dist.log_prob(a)
            entropy = pi_a_dist.entropy() * entropy_coef
            
            ratio = torch.exp(pi_a - prob_a.detach())
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss_first = (-torch.min(surr1, surr2) - entropy).mean() 
            loss_second = critic_coef * F.smooth_l1_loss(self.v(s) , td_target.detach())
            loss = loss_first + loss_second
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
entropy_coef = 1e-2
critic_coef = 1
learning_rate = 0.0001
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

env = gym.make('Pendulum-v0')
model = PPO()

print_interval = 10

def main(render = False):
    score = 0.0
    global_step = 0
    for n_epi in range(10000):
        
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                global_step += 1 
                if render:    
                    env.render()
                prob,mu,sigma = model.pi(torch.from_numpy(s).float())
                old_log_prob = torch.distributions.Normal(mu,sigma).log_prob(prob).detach().item()
                action = prob.item()
                s_prime, r, done, info = env.step([action])
    
                model.put_data((s, action, r/100.0, s_prime, \
                                old_log_prob, done))
                s = s_prime
                
                score += r
                if done:
                    break
            model.train_net()
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

main()