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

def MC_gradient_descent():
    
    grid_size = 5
    n_states = grid_size * grid_size
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    n_actions = len(actions)
    terminal_states = [0, n_states - 1]
    gamma = 1.0
    alpha = 0.01
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

    # --- Features vector one-hot for (state, action) ---
    def phi_sa(s, a_idx):
        vec = np.zeros(n_states * n_actions)
        vec[s * n_actions + a_idx] = 1.0
        return vec

    # --- Pesos Q(s,a) ---
    weights = np.zeros(n_states * n_actions)

    for ep in range(n_episodes):
        state = np.random.randint(n_states)
        episode = []
        done = False

        # Generar episodio siguiendo política aleatoria
        while not done:
            a_idx = np.random.randint(n_actions)
            action = actions[a_idx]
            next_state, reward, done = step(state, action)
            episode.append((state, a_idx, reward))
            state = next_state

        # Calcular retornos hacia atrás y actualizar w (first-visit MC)
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a_idx, r = episode[t]
            G = r + gamma * G
            if (s, a_idx) not in visited:
                visited.add((s, a_idx))
                Q_hat = weights @ phi_sa(s, a_idx)
                weights += alpha * (G - Q_hat) * phi_sa(s, a_idx)

    # --- Resultado final ---
    Q = weights.reshape((n_states, n_actions))
    
    # Visualización: podemos mostrar V(s) = max_a Q(s,a)
    V = Q.max(axis=1).reshape((grid_size, grid_size))
    plt.imshow(V, cmap="viridis")
    plt.colorbar(label="V(s) estimado (max_a Q(s,a))")
    plt.title("Linear Monte Carlo Q-function Approximation (Política Aleatoria)")
    plt.show()

    print("Valores estimados V(s) = max_a Q(s,a):")
    print(np.round(V, 2))
    print("Valores Q(s,a) (reshape n_states x n_actions):")
    print(np.round(Q, 2))

def Q_learning_Qfa_gradient_descent_control():
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

