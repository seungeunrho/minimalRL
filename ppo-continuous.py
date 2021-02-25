import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
entropy_coef = 1e-2
critic_coef = 1
learning_rate = 0.0003
gamma         = 0.9
lmbda         = 0.9
eps_clip      = 0.2
K_epoch       = 10
T_horizon     = 20


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(3,64)
        self.fc2   = nn.Linear(64,256)
        self.fc_v  = nn.Linear(256,1)
        self.fc_pi = nn.Linear(256,1)
        self.fc_sigma = nn.Linear(256,1)
    
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = 2 * F.tanh(self.fc_pi(x))
        sigma = F.softplus(self.fc_sigma(x)) +1e-3

        return mu,sigma
        
    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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
        s, a, r, s_prime, done_mask, old_log_prob = self.make_batch()
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

            curr_mu,curr_sigma = self.pi(s)
            
            curr_dist = torch.distributions.Normal(curr_mu,curr_sigma)
            curr_log_prob = curr_dist.log_prob(a)
            entropy = curr_dist.entropy() * entropy_coef
            
            ratio = torch.exp(curr_log_prob - old_log_prob.detach())
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            actor_loss = (-torch.min(surr1, surr2) - entropy).mean() 
            critic_loss = critic_coef * F.smooth_l1_loss(self.v(s).float() , td_target.detach().float())
            loss = actor_loss + critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            

env = gym.make('Pendulum-v0')
model = PPO()

print_interval = 20

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
                mu,sigma = model.pi(torch.from_numpy(s).float())
                dist = torch.distributions.Normal(mu,sigma)
                
                action = dist.sample()
                log_prob = dist.log_prob(action)
                s_prime, r, done, info = env.step([action.item()])
    
                model.put_data((s, action, r/10.0, s_prime, \
                                log_prob, done))
                s = s_prime
                
                score += r
                if done:
                    break
            model.train_net()
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

main()
