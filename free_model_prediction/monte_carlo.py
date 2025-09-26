# -*- coding: utf-8 -*-
"""
Created on 2025

@author: andres.sanchez
"""

import random
import numpy as np
from collections import defaultdict

def MC_grid(grid_size_x, grid_size_y,
             N_EPISODES = 100,
             N_STEPS = 7, goal = (3,3),
             start = (0,0)):
    
    S = [(m, n) for m in range(grid_size_x) for n in range(grid_size_y)]
    V = {s:0 for s in S}
    A = [(1,0), (0,1), (-1,0), (0,-1)]

    GAMMA = 0.9
    ALPHA = 0.1

    n_episodes = 1000
    n_steps = 10

    def softmax_pick_action(s, V, tau=1.0, grid_size=(4,4)):
        
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

        prefs = np.array(prefs)
        exp_prefs = np.exp((prefs - np.max(prefs)) / tau)
        probs = exp_prefs / np.sum(exp_prefs)
        
        chosen_action = valid_actions[np.random.choice(len(valid_actions), p=probs)]
            
        return chosen_action
        
    def step(a, s, tau = 0.1):
        
        done = False
        
        s_prima_x = s[0] + a[0]
        s_prima_y = s[1] + a[1]
        
        s_prima = (s_prima_x, s_prima_y)
        
        if s_prima == goal:
            r = 0
            done = True
        else:
            r = -1
        
        return r, s_prima, done
        
    for ep in range(n_episodes):
        if ep % 100 == 0:
            print(f'Episode {ep}')
        s = start
        episode = []
        for st in range(1,n_steps):
            a = softmax_pick_action(s, V)
            r, s_prima, done = step(a, s)
            episode.append((s, r))
            s = s_prima
            if done:
                break
        
        G = 0
        for (s, r) in reversed(episode):
            G = r + GAMMA * G
            V[s] += ALPHA * (G - V[s])  # MC update
        
    grid_values = np.zeros((grid_size_x, grid_size_y))
    for (x, y), value in V.items():
        grid_values[x, y] = value    
     
    print(f'It took {ep} episodes')
    print(grid_values)           
       
