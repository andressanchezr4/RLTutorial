# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 17:07:17 2025

@author: andres.sanchez
"""

import numpy as np
import matplotlib.pyplot as plt

def TD0_gradient_descent():
    grid_size = 5
    n_states = grid_size * grid_size
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # derecha, izquierda, abajo, arriba
    terminal_states = [0, n_states - 1]  # esquinas (0,0) y (4,4)
    gamma = 1.0
    alpha = 0.01
    n_episodes = 1000
    
    def state_to_xy(s):
        return divmod(s, grid_size)
    
    def xy_to_state(x, y):
        return x * grid_size + y
    
    def step(state, action):
        """Ejecuta una acción en el grid, retorna next_state, reward, done"""
        if state in terminal_states:
            return state, 0, True  # si estamos en terminal, no avanzamos
    
        x, y = state_to_xy(state)
        dx, dy = action
        
        nx = max(0, min(grid_size - 1, x + dx))
        ny = max(0, min(grid_size - 1, y + dy))
        
        next_state = xy_to_state(nx, ny)
        
        if next_state in terminal_states:
            reward = 0 
        else:
            reward = -1  # penalizamos cada paso
        done = next_state in terminal_states
        return next_state, reward, done
    
    # --- Features vector one-hot ---
    def phi(state):
        vec = np.zeros(n_states)
        vec[state] = 1.0
        return vec
    
    weights = np.zeros(n_states)
    
    for ep in range(n_episodes):
        state = np.random.randint(n_states)
        done = False
    
        while not done:
            action = actions[np.random.randint(len(actions))]  # política aleatoria
            next_state, reward, done = step(state, action)
    
            V_s = weights @ phi(state)
            V_next = weights @ phi(next_state)
            
            td_target = reward + gamma * V_next
            td_error = td_target - V_s
    
            weights += alpha * td_error * phi(state)
            state = next_state
    
    # --- Resultado final ---
    V = weights.reshape((grid_size, grid_size))
    
    plt.imshow(V, cmap="viridis")
    plt.colorbar(label="V(s) estimado")
    plt.title("TD(0) Value Function Approximation (Política Aleatoria)")
    plt.show()
    
    print("Valores estimados V(s):")
    print(np.round(V, 2))


def TD0_lambda_gradient_descent(lmbda=0.8):
    grid_size = 5
    n_states = grid_size * grid_size
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # derecha, izquierda, abajo, arriba
    terminal_states = [0, n_states - 1]  # esquinas (0,0) y (4,4)
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

    # --- Features vector one-hot ---
    def phi(state):
        vec = np.zeros(n_states)
        vec[state] = 1.0
        return vec

    weights = np.zeros(n_states)

    for ep in range(n_episodes):
        state = np.random.randint(n_states)
        done = False
        e = np.zeros(n_states)  # trazas de elegibilidad inicializadas en cero

        while not done:
            action = actions[np.random.randint(len(actions))]  # política aleatoria
            next_state, reward, done = step(state, action)

            V_s = weights @ phi(state)
            V_next = weights @ phi(next_state)
            td_error = reward + gamma * V_next - V_s

            # Actualización de trazas de elegibilidad
            e = gamma * lmbda * e + phi(state)

            # Actualización de pesos
            weights += alpha * td_error * e

            state = next_state

    # --- Resultado final ---
    V = weights.reshape((grid_size, grid_size))
    plt.imshow(V, cmap="viridis")
    plt.colorbar(label="V(s) estimado")
    plt.title(f"TD({lmbda}) Value Function Approximation (Política Aleatoria)")
    plt.show()
