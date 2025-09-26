# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 17:35:53 2025

@author: andres.sanchez
"""

import random
import numpy as np
from collections import defaultdict

def TD0_line(line_size, n_episodes = 1000, 
             max_steps = 100):
    
    def step(s, a):
        s_prima = max(0, min(line_size, s + a))
        if s_prima == goal:
            r = 0 
        else:
            r = -1
        done = (s_prima == goal)
        return s_prima, r, done
    
    # Policy: epsilon-greedy around "always go right" baseline, but you can fix it
    # def epsilon_greedy_policy(s, V, epsilon=0.1):
    #     # simple example: prefer +1 if value to the right is higher (1-step lookahead)
    #     if random.random() < epsilon:
    #         return random.choice(A)
    #     if s < goal:
    #         return 1
    #     else: 
    #         return -1
    
    def softmax_policy(s, V, tau=1.0):
        """
        Softmax over one-step lookahead values.
        tau (temperature) controls exploration:
          - high tau -> more random
          - low tau  -> more greedy
        """
        prefs = []
        for a in A:
            s_prima = max(0, min(line_size, s + a))
            prefs.append(V[s_prima])  # preference = value of next state
        
        prefs = np.array(prefs)
        # numerically stable softmax
        exp_prefs = np.exp((prefs - np.max(prefs)) / tau)
        probs = exp_prefs / np.sum(exp_prefs)
        return np.random.choice(A, p=probs)
    
    goal = line_size
    S = list(range(line_size + 1))
    A = [-1, 1]
    
    alpha = 0.1
    gamma = 0.9
    
    V = {s: 0.0 for s in S}
    
    for ep in range(n_episodes):
        s = 0
        if ep % 100 == 0:
            print(f'Episode: {ep}')
        for t in range(max_steps):
            # a = epsilon_greedy_policy(s, V, epsilon=0.1)
            a = softmax_policy(s, V)
            s_prima, r, done = step(s, a)
            # TD(0) update
            if done:
                v_prima = 0
            else:
                v_prima = V[s_prima]

            V[s] += alpha * (r + gamma * v_prima - V[s])
            
            s = s_prima
            if done:
                break
         
    vals = np.array([V[s] for s in S])
    print(f'It took {ep} episodes')
    print("TD(0) state values:", np.round(vals, 3))


def TD0_gridw_slip(grid_size_x, grid_size_y,
             N_EPISODES = 100,
             N_STEPS = 7, goal = (3,3),
             start = (0,0)):
    
    perpendicular = {
        (0,1): [(-1,0), (1,0)],   # Si eliges ir a la derecha, puedes desviarte hacia arriba o abajo
        (1,0): [(0,1), (0,-1)],   # Si eliges ir abajo, puedes desviarte hacia derecha o izquierda
        (-1,0): [(0,-1), (0,1)],  # Si eliges ir arriba, puedes desviarte hacia izquierda o derecha
        (0,-1): [(1,0), (-1,0)]   # Si eliges ir a la izquierda, puedes desviarte hacia abajo o arriba
    }

    S = [(n,m) for n in range(grid_size_x) for m in range(grid_size_y)]
    V = {s: 0 for s in S}
    A = [(0,1), (1,0), (-1,0), (0,-1)]
    
    alpha = 0.1
    gamma = 0.9
    slip_prob = 0.1
    
    def softmax_policy(s, V, tau=1.0, grid_size=(4,4)):
        
        x, y = s
        nx, ny = grid_size
        prefs = []
        valid_actions = []
        
        for a in A:
            dx, dy = a
            nx_, ny_ = x + dx, y + dy
            
            # ignorar movimientos que salen fuera de la grid
            if 0 <= nx_ < nx and 0 <= ny_ < ny:
                prefs.append(V.get((nx_, ny_), 0.0))
                valid_actions.append(a)
        
        # convertir a array
        prefs = np.array(prefs)
        exp_prefs = np.exp((prefs - np.max(prefs)) / tau)
        probs = exp_prefs / np.sum(exp_prefs)
        
        chosen_action = valid_actions[np.random.choice(len(valid_actions), p=probs)]
        
        return chosen_action
            
    def step(s, a):
        
        done = False
        s_prima_x = s[0] + a[0]
        s_prima_y = s[1] + a[1]
        s_prima = (s_prima_x, s_prima_y)

        # here is the slip 
        if random.random() < slip_prob:
            left, right = perpendicular[a]
            for dx_p, dy_p in [left, right]:
                next_x = min(max(s_prima_x + dx_p, 0), grid_size_x - 1)
                next_y = min(max(s_prima_y + dy_p, 0), grid_size_y - 1)
                s_prima = (next_x, next_y)
                
        if s_prima == goal:
            r = 0
            done = True
        else:
            r = -1
            
        return s_prima, r, done
         
        
    for e in range(N_EPISODES):
        s = start
        if e % 100 == 0:
            print(f'Episode: {e}')
        for st in range(N_STEPS):
            a = softmax_policy(s, V)
            s_prima, r, done = step(s, a)
            if done:
                v_prima = 0
            else:
                v_prima = V[s_prima]
            
            V[s] += alpha * (r + gamma * v_prima - V[s])
            s = s_prima
            
            if done:
                break
    
    grid_values = np.zeros((grid_size_x, grid_size_y))
    for (x, y), value in V.items():
        grid_values[x, y] = value    
     
    print(f'It took {e} episodes')
    print(grid_values)    


def TD0_lambda_grid(grid_size_x, grid_size_y,
             N_EPISODES = 100,
             N_STEPS = 7, goal = (3,3),
             start = (0,0)):

    S = [(m, n) for m in range(grid_size_x) for n in range(grid_size_y)]
    V = {s:0 for s in S}
    A = [(1,0), (0,1), (-1,0), (0,-1)]
    
    GAMMA = 0.9
    ALPHA = 0.1
    LAMBDA = 0.8   # <-- aquí está el λ, si = 0 TD0, si = 1 MC
    
    n_episodes = 1000
    n_steps = 20
    
    def softmax_pick_action(s, V, tau=1.0, grid_size=(4,4)):
        x, y = s
        nx, ny = grid_size
        prefs = []
        valid_actions = []
        for a in A:
            dx, dy = a
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < nx and 0 <= ny_ < ny:
                prefs.append(V.get((nx_, ny_), 0.0))
                valid_actions.append(a)
        prefs = np.array(prefs)
        exp_prefs = np.exp((prefs - np.max(prefs)) / tau)
        probs = exp_prefs / np.sum(exp_prefs)
        chosen_action = valid_actions[np.random.choice(len(valid_actions), p=probs)]
        return chosen_action
    
    def step(a, s):
        s_prima = (s[0] + a[0], s[1] + a[1])
        done = s_prima == goal
        r = 0 if done else -1
        return r, s_prima, done
    
    # Por cada episodio, lambda (Forward view) DISMINUYE la influencia de dicho estado 
    # si se se visita mas tarde y el elegibility trace (Backwar view, E) calcula un peso para
    # cada estado en funcion del numero de veces que se visita
    for ep in range(n_episodes):
        if ep % 100 == 0:
            print(f"Episode {ep}")  
        s = start
        E = {state: 0 for state in S} # Nos calcula un peso para cada estado en cada episodio
        
        for st in range(n_steps):
            a = softmax_pick_action(s, V)
            r, s_prima, done = step(a, s)

            delta = r + GAMMA * V[s_prima] - V[s]
    
            # Incrementar eligibility del estado actual
            E[s] += 1
    
            # Actualizar todos los estados con elegibility > 0
            for state in S:
                V[state] += ALPHA * delta * E[state]
                E[state] *= GAMMA * LAMBDA  # decaimiento de trace
    
            s = s_prima
            if done:
                break
    
    grid_values = np.zeros((grid_size_x, grid_size_y))
    for (x, y), value in V.items():
        grid_values[x, y] = value
    
    print("Valores finales con TD(λ):")
    print(grid_values)
    
    
    
