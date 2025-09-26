# -*- coding: utf-8 -*-
"""
Created on 2025

@author: andres.sanchez
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

def Q_learning_gradient_descent():
    grid_size = 5
    n_states = grid_size * grid_size
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    n_actions = len(actions)
    terminal_states = [0, n_states - 1]
    gamma = 1.0
    alpha = 0.1
    epsilon = 0.1
    n_episodes = 1000

    def state_to_xy(s):
        return divmod(s, grid_size)

    def xy_to_state(x, y):
        return x * grid_size + y

    def step(state, action):
        if state in terminal_states:
            return state, 0, True
        x, y = state_to_xy(state)
        dx, dy = action
        nx = max(0, min(grid_size - 1, x + dx))
        ny = max(0, min(grid_size - 1, y + dy))
        next_state = xy_to_state(nx, ny)
        reward = 0 if next_state in terminal_states else -1
        done = next_state in terminal_states
        return next_state, reward, done

    # --- One-hot feature vector for (s,a) ---
    def phi_sa(s, a_idx):
        vec = np.zeros(n_states * n_actions)
        vec[s * n_actions + a_idx] = 1.0
        return vec

    # --- Initialize weights ---
    weights = np.zeros(n_states * n_actions)

    for ep in range(n_episodes):
        state = np.random.randint(n_states)
        done = False

        while not done:
            # --- ε-greedy policy selection ---
            q_values = [weights @ phi_sa(state, a) for a in range(n_actions)]
            if np.random.rand() < epsilon:
                a_idx = np.random.randint(n_actions)
            else:
                a_idx = np.argmax(q_values)

            action = actions[a_idx]
            next_state, reward, done = step(state, action)

            # --- TD target using max over next actions (Q-learning) ---
            q_hat = weights @ phi_sa(state, a_idx)
            if done:
                target = reward
            else:
                q_next = max(weights @ phi_sa(next_state, a_next) for a_next in range(n_actions))
                target = reward + gamma * q_next

            # --- Gradient descent update ---
            weights += alpha * (target - q_hat) * phi_sa(state, a_idx)

            state = next_state

    # --- Reshape to Q-table and compute V(s) ---
    Q = weights.reshape((n_states, n_actions))
    V = Q.max(axis=1).reshape((grid_size, grid_size))

    # --- Visualization ---
    plt.imshow(V, cmap="viridis")
    plt.colorbar(label="V(s) = max_a Q(s,a)")
    plt.title("Q-learning with Linear Function Approximation")
    plt.show()

    print("Estimated V(s) = max_a Q(s,a):")
    print(np.round(V, 2))
    print("Q(s,a) table:")
    print(np.round(Q, 2))

def Deep_Q_learning():
    grid_size = 5
    n_states = grid_size * grid_size
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    n_actions = len(actions)
    terminal_states = [0, n_states - 1]
    gamma = 1.0
    alpha = 1e-3           # learning rate
    epsilon = 0.1
    n_episodes = 2000
    batch_size = 32
    memory_capacity = 5000

    def state_to_xy(s):
        return divmod(s, grid_size)

    def xy_to_state(x, y):
        return x * grid_size + y

    def step(state, action):
        if state in terminal_states:
            return state, 0, True
        x, y = state_to_xy(state)
        dx, dy = action
        nx = max(0, min(grid_size - 1, x + dx))
        ny = max(0, min(grid_size - 1, y + dy))
        next_state = xy_to_state(nx, ny)
        reward = 0 if next_state in terminal_states else -1
        done = next_state in terminal_states
        return next_state, reward, done

    # --- Neural network Q(s) -> [Q(s,a) for all a] ---
    class QNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(n_states, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions)
            )
        def forward(self, x):
            return self.fc(x)

    # Initialize network and optimizer
    q_net = QNetwork()
    optimizer = optim.Adam(q_net.parameters(), lr=alpha)
    loss_fn = nn.MSELoss()

    # Replay buffer
    memory = deque(maxlen=memory_capacity)

    def one_hot_state(s):
        vec = np.zeros(n_states)
        vec[s] = 1.0
        return vec

    for ep in range(n_episodes):
        state = np.random.randint(n_states)
        done = False
        while not done:
            state_vec = torch.tensor(one_hot_state(state), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = q_net(state_vec).numpy().flatten()
            
            # ε-greedy action
            if random.random() < epsilon:
                a_idx = np.random.randint(n_actions)
            else:
                a_idx = np.argmax(q_values)

            action = actions[a_idx]
            next_state, reward, done = step(state, action)

            memory.append((state, a_idx, reward, next_state, done))
            state = next_state

            # Train only if enough samples
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions_idx, rewards, next_states, dones = zip(*batch)

                states_tensor = torch.tensor([one_hot_state(s) for s in states], dtype=torch.float32)
                next_states_tensor = torch.tensor([one_hot_state(ns) for ns in next_states], dtype=torch.float32)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                actions_tensor = torch.tensor(actions_idx, dtype=torch.int64)
                dones_tensor = torch.tensor(dones, dtype=torch.bool)

                q_values = q_net(states_tensor)
                q_sa = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    q_next = q_net(next_states_tensor).max(1)[0]
                    target = rewards_tensor + gamma * q_next * (~dones_tensor)

                loss = loss_fn(q_sa, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # --- Compute V(s) from network ---
    all_states = torch.eye(n_states)
    with torch.no_grad():
        Q_all = q_net(all_states).numpy()
    V = Q_all.max(axis=1).reshape((grid_size, grid_size))

    plt.imshow(V, cmap="viridis")
    plt.colorbar(label="V(s) = max_a Q(s,a)")
    plt.title("Deep Q-learning Value Function (DQN)")
    plt.show()

    print("Estimated V(s):")
    print(np.round(V, 2))
    print("Q(s,a) examples (first few states):")
    print(np.round(Q_all[:5], 2))

