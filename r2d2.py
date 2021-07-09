import random
from collections import deque, namedtuple

import gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32
cell_size = 64
sequence_length = 20
over_lapping_length = 10
burn_in_length = 10
Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask', 'rnn_state'))

class Qnet(nn.Module):
    def __init__(self, num_inputs=4, num_outputs=2):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=16, batch_first=True)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, num_outputs)

    def forward(self, x, hidden=None):
        # [batch_size, sequence_length, num_inputs]
        if len(x.shape) == 1:
            batch_size = 1
            sequence_length = 1
            x = x.view(batch_size, sequence_length, -1)
        else:
            batch_size = x.size()[0]
            sequence_length = x.size()[1]
        out, hidden = self.lstm(x, hidden)

        out = F.relu(self.fc1(out))
        qvalue = self.fc2(out).view(batch_size, sequence_length, self.num_outputs)

        return qvalue, hidden


class LocalBuffer(object):
    def __init__(self):
        self.local_memory = []
        self.memory = []
        self.over_lapping_from_prev = []

    def push(self, state, next_state, action, reward, mask, rnn_state):
        self.local_memory.append(
            Transition(state, next_state, action, reward, mask, torch.stack(rnn_state).view(2, -1)))
        if (len(self.local_memory) + len(self.over_lapping_from_prev)) == sequence_length or mask == 0:
            self.local_memory = self.over_lapping_from_prev + self.local_memory
            length = len(self.local_memory)
            while len(self.local_memory) < sequence_length:  # zero padding to standardize length of the each experience
                self.local_memory.append(Transition(torch.tensor([.0, .0, .0, .0], dtype=torch.float), torch.tensor([.0, .0, .0, .0], dtype=torch.float), 0, 0, 0,
                                                    torch.zeros([2, 1, 16]).view(2, -1)))  # rnn state = [(hidden,cell), seq_, dim]
            self.memory.append([self.local_memory, length])  # length tells true length of the memory
            if mask == 0:
                self.over_lapping_from_prev = []
            else:
                self.over_lapping_from_prev = self.local_memory[over_lapping_length:]
            self.local_memory = []

    def get(self):
        episodes = self.memory
        batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_rnn_state = [], [], [], [], [], []
        lengths = []
        for episode, length in episodes:
            batch = Transition(*zip(*episode))

            batch_state.append(torch.stack(list(batch.state)))
            batch_next_state.append(torch.stack(list(batch.next_state)))
            batch_action.append(torch.tensor(list(batch.action)))
            batch_reward.append(torch.tensor(list(batch.reward)))
            batch_mask.append(torch.tensor(list(batch.mask)))
            batch_rnn_state.append(torch.stack(list(batch.rnn_state)))
            lengths.append(length)

        self.memory = []
        return Transition(batch_state, batch_next_state, batch_action, batch_reward, batch_mask,
                          batch_rnn_state), lengths


class Memory(object):
    def __init__(self):
        self.memory = deque(maxlen=buffer_limit)

    def size(self):
        return len(self.memory)

    def put(self, batch, lengths):
        for i in range(len(batch)):
            self.memory.append([Transition(batch.state[i], batch.next_state[i], batch.action[i], batch.reward[i],
                                           batch.mask[i], batch.rnn_state[i]), lengths[i]])

    def sample(self, batch_size):
        indexes = np.random.choice(range(len(self.memory)), batch_size)
        episodes = [self.memory[idx][0] for idx in indexes]
        lengths = [self.memory[idx][1] for idx in indexes]

        batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_step, batch_rnn_state = [], [], [], [], [], [], []
        for episode in episodes:
            batch_state.append(episode.state)
            batch_next_state.append(episode.next_state)
            batch_action.append(episode.action)
            batch_reward.append(episode.reward)
            batch_mask.append(episode.mask)
            batch_rnn_state.append(episode.rnn_state)

        return Transition(batch_state, batch_next_state, batch_action, batch_reward, batch_mask,
                          batch_rnn_state), indexes, lengths


def learner_process(model, target_model, exp_q, lock):
    leaner = Learner(model, target_model, exp_q, lock)
    leaner.run()


class Learner:
    def __init__(self, model, target_model, share_exp_mem, lock):
        self.q = model
        self.q_target = target_model
        self.optimizer = optim.Adam(self.q.parameters())
        self.share_exp_mem = share_exp_mem
        self.lock = lock

        self.n_epochs = 0

    def run(self):
        while True:
            if self.share_exp_mem.size() > batch_size:
                batch, indexes, lengths = self.share_exp_mem.sample(batch_size)
                self.train(batch, lengths)
                self.n_epochs += 1
                if self.n_epochs % 5 == 0:
                    self.q_target.load_state_dict(self.q.state_dict())

    def train(self, batch, lengths):
        def slice_burn_in(item):
            return item[:, burn_in_length:, :]

        batch_size = torch.stack(batch.state).size()[0]
        states = torch.stack(batch.state).view(batch_size, sequence_length, self.q.num_inputs)
        next_states = torch.stack(batch.next_state).view(batch_size, sequence_length, self.q.num_inputs)
        actions = torch.stack(batch.action).view(batch_size, sequence_length, -1).long()
        rewards = torch.stack(batch.reward).view(batch_size, sequence_length, -1)
        masks = torch.stack(batch.mask).view(batch_size, sequence_length, -1)
        rnn_state = torch.stack(batch.rnn_state).view(batch_size, sequence_length, 2, -1)

        [h0, c0] = rnn_state[:, 0, :, :].transpose(0, 1)  # the first hidden state among sequence_length
        h0 = h0.unsqueeze(0).detach()
        c0 = c0.unsqueeze(0).detach()

        [h1, c1] = rnn_state[:, 1, :, :].transpose(0, 1)  # the second hidden state among sequence_length
        h1 = h1.unsqueeze(0).detach()
        c1 = c1.unsqueeze(0).detach()

        pred, _ = self.q(states, (h0, c0))
        next_pred, _ = self.q_target(next_states, (h1, c1))

        next_pred_online, _ = self.q(next_states, (h1, c1))

        pred = slice_burn_in(pred)
        next_pred = slice_burn_in(next_pred)
        actions = slice_burn_in(actions)
        rewards = slice_burn_in(rewards)
        masks = slice_burn_in(masks)
        next_pred_online = slice_burn_in(next_pred_online)

        pred = pred.gather(2, actions)

        _, next_pred_online_action = next_pred_online.max(2)

        target = rewards + masks * gamma * next_pred.gather(2, next_pred_online_action.unsqueeze(2))

        td_error = pred - target.detach()

        for idx, length in enumerate(lengths):
            td_error[idx][length - burn_in_length:][:] = 0

        loss = pow(td_error, 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def actor_process(actor_id, n_actors, model, target_model, exp_q, lock):
    actor = Actor(actor_id, n_actors, model, target_model, exp_q, lock)
    actor.run()


class Actor:
    def __init__(self, actor_id, n_actors, model, target_model, share_exp_mem, lock):
        self.env = gym.make('CartPole-v1')
        self.actor_id = actor_id
        self.epsilon = 0.1 + (actor_id / 7) / n_actors  # 0.4 ** (1 + actor_id * 7 / (n_actors - 1))

        self.local_buffer = LocalBuffer()
        self.q = model
        self.q_target = target_model
        self.net_load_interval = 10
        self.overlap_length = 5

        self.share_exp_mem = share_exp_mem
        self.lock = lock

    def run(self):
        for e in range(30000):
            done = False

            score = 0
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float)
            hidden = (torch.zeros(1, 1, 16), torch.zeros(1, 1, 16))

            while not done:
                epsilon = max(0.01, self.epsilon - 0.01 * (e / 200))  # Linear annealing from 8% to 1%
                with torch.no_grad():
                    q_value, new_hidden = self.q(state)
                    target_q_value, target_new_hidden = self.q_target(state)  #todo : replay 에 target hidden 도 넣어야 할텐데..?
                if random.random() < epsilon:
                    action = random.randint(0, 1)
                else:
                    action = q_value.argmax().item()

                next_state, reward, done, _ = self.env.step(action)

                # next_state = state_to_partial_observability(next_state)
                next_state = torch.tensor(next_state, dtype=torch.float)

                mask = 0 if done else 1
                self.local_buffer.push(state, next_state, action, reward, mask,
                                       hidden)  # todo : target_hidden 은 저장 안하나?
                hidden = new_hidden
                if len(self.local_buffer.memory) == batch_size:
                    batch, lengths = self.local_buffer.get()
                    # memory.push(td_error, batch, lengths)
                    self.lock.acquire()
                    self.share_exp_mem.put(batch, lengths)
                    self.lock.release()

                score += reward
                state = next_state

            if e % 20 == 0:
                print('episodes:', e, 'actor_id:', self.actor_id, 'reward:', score)


def main():
    model = Qnet()
    target_model = Qnet()
    target_model.load_state_dict(model.state_dict())
    model.share_memory()
    target_model.share_memory()

    mp.Manager().register('Memory', Memory)
    manager = mp.Manager()
    experience_memory = manager.Memory()

    l = mp.Lock()

    # learner process
    processes = [mp.Process(
        target=learner_process,
        args=(model, target_model, experience_memory, l))]

    # actor process
    n_actors = 2
    for actor_id in range(n_actors):
        processes.append(mp.Process(
            target=actor_process,
            args=(actor_id, n_actors, model, target_model, experience_memory, l)))

    for i in range(len(processes)):
        processes[i].start()

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
