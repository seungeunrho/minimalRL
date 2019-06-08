import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import time

#Hyperparameters
n_train_processes = 3
learning_rate     = 0.0002
update_interval   = 5
gamma             = 0.98
max_train_ep      = 300
max_test_ep       = 400

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v = nn.Linear(256,1)
        
    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

def train(model, rank):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    env = gym.make('CartPole-v1')

    for n_epi in range(max_train_ep):
        done = False
        s = env.reset()
        while not done:
            s_lst, a_lst, r_lst = [], [], []
            for t in range(update_interval):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                
                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r/100.0)
                
                s = s_prime               
                if done:
                    break
                                        
            R = 0.0
            R_lst = []
            for reward in r_lst[::-1]:
                R = gamma * R + reward
                R_lst.append([R])
            R_lst.reverse()
            
            done_mask = 0.0 if done else 1.0
            s_batch, a_batch, R_batch, s_final = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                                 torch.tensor(R_lst), torch.tensor(s_prime, dtype=torch.float)

            td_target = R_batch + gamma * model.v(s_final) * done_mask
            advantage = td_target - model.v(s_batch)
            pi = model.pi(s_batch,softmax_dim=1)
            pi_a = pi.gather(1,a_batch)
            loss = -torch.log(pi_a) * advantage.detach() + F.smooth_l1_loss(model.v(s_batch), td_target.detach())
            
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()    

    env.close()
    print("Training process {} reached maximum episode.".format(rank))

def test(model):
    env = gym.make('CartPole-v1')
    score = 0.0
    print_interval = 20
    
    for n_epi in range(max_test_ep):
        done = False
        s = env.reset()
        while not done:
            prob = model.pi(torch.from_numpy(s).float())
            a = Categorical(prob).sample().item()
            s_prime, r, done, info = env.step(a)
            s = s_prime
            score += r
            
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
            time.sleep(1)
    env.close()

if __name__ == '__main__':
    model = ActorCritic()
    model.share_memory()
    processes = []
    for rank in range(n_train_processes + 1): # + 1 for test process
        if rank == 0:
            p = mp.Process(target=test, args=(model,))
        else:           
            p = mp.Process(target=train, args=(model, rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()