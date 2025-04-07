import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class Actor(nn.Module):
    def __init__(self, input_channels, output_size):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(2,1), padding=0, stride=1)
        self.fc1 = nn.Linear(32 * 9, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        actions = torch.sigmoid(self.fc4(x))
        return actions

class Critic(nn.Module):
    def __init__(self, input_channels, action_size):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(2,1), padding=0, stride=1)
        self.fc1 = nn.Linear((32 * 9) + action_size, 512)  # 添加action_size作为输入维度
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.relu(self.conv1(state))
        x = x.view(x.size(0), -1)
        x = torch.cat((x, action), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        value = self.fc4(x)
        return value

class ACLearner:
    def __init__(self, input_channels, action_size, learning_rate=0.001, gamma=0.9, epsilon=0.95,
                 memory_size=5000, batch_size=10, replace_target_iter=300):
        self.input_channels = input_channels
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_initial = epsilon
        self.min_epsilon = 0.02
        self.decay_rate = (self.min_epsilon / self.epsilon_initial) ** (1 / 80)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(input_channels, action_size).to(self.device)
        self.critic = Critic(input_channels, action_size).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()
        self.memory = deque(maxlen=self.memory_size)
        self.learn_step_counter = 0

    def select_action(self, state):
        temp_ac = random.random()
        if temp_ac < self.epsilon:
            random_integer = random.randint(1, 9)
            ones = [1] * random_integer
            zeros = [0] * (9 - random_integer)
            radmaction = ones + zeros
            random.shuffle(radmaction)

            radmaction = torch.FloatTensor(np.array(radmaction)).to(self.device)
            self.update_epsilon()
            number_to_write = 111
            number_str = str(number_to_write) + '\n'
            with open('./res_action.txt', 'a') as file:
                file.write(number_str)
            return radmaction
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            actions = self.actor(state)
            self.update_epsilon()
            number_to_write = 222
            number_str = str(number_to_write) + '\n'
            with open('./res_action.txt', 'a') as file:
                file.write(number_str)
            return actions.flatten()

    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay_rate
            self.epsilon = max(self.min_epsilon, self.epsilon)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_experience(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def train(self):
        batch_size = self.batch_size
        if len(self.memory) < self.batch_size:
            batch_size = len(self.memory)
        states, actions, rewards, next_states, dones = self.sample_experience(batch_size)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        values = self.critic(states, actions)
        next_values = self.critic(next_states, self.actor(next_states)).detach()
        rewards = rewards.unsqueeze(1)
        expected_values = rewards + self.gamma * next_values

        critic_loss = self.loss_function(values, expected_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action_preds = self.actor(states)
        actor_loss = -torch.mean(self.critic(states, action_preds))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def Initialize_epsilon(self, num_epsilon, num_batchsize):
        self.epsilon = num_epsilon
        self.batch_size = num_batchsize + self.batch_size
