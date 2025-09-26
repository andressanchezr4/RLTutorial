# -*- coding: utf-8 -*-
"""
Created on 2025

@author: andres.sanchez
"""

from collections import defaultdict
import numpy as np 
import math

def markov_reward_process_line(line_size):
    
    goal = line_size
    S = [n for n in range(line_size+1)]
    A = [-1, 1] # ojo con las acciones que no hacen nada
    V = {n:0 for n in range(line_size+1)}
    
    THETA = 0.01
    GAMMA = 0.9
    
    P = defaultdict(list)
    for s in S:
        if s == goal:
            for a in A:
                P[s].append((1, s, 0)) # ojo con las rewards que no son 0 que pueden causar un infinite loop 
        else:
            for a in A:
                s_prima = s + a
                s_prima = max(0, min(line_size, s + a))
                if s_prima == goal:
                    r = 0
                else: 
                    r = -1
                P[s].append((1/len(A), s_prima, r))
    
    delta = 0 
    n_iterations = 0   
    while delta >= THETA or n_iterations == 0:  
        delta = 0
        for s in S:
            old_v = V[s]
            v = 0
            for my_tuple in P[s]:
                prob, s_prima, r = my_tuple
                v += prob * (r + GAMMA * V[s_prima])
            V[s] = v
            delta = max(delta, abs(old_v - v))
        print(n_iterations, delta)
        n_iterations += 1
    
    grid_values = np.zeros(line_size+1)
    for x, value in V.items():
        grid_values[x] = value         
    
    print('Final V:') 
    print(grid_values) 


def markov_reward_process_gridw(grid_size_x, 
                          grid_size_y, 
                          goal):
    
    THETA = 0.01
    GAMMA = 0.9
    A = [(0,1), (1,0), (-1,0), (0,-1)]
    S = [(n,m) for n in range(grid_size_x) for m in range(grid_size_y)]
    V = {s:0 for s in S}
    
    # Model simulation (Transition dynamics + rewards)
    P = defaultdict(list)
    for s in S:
        if s == goal:
            P[s].append((1.0, s, 0))
        else:
            for a in A:
                x, y = s
                dx, dy = a
                
                next_x = min(max(x + dx, 0), grid_size_x - 1) # min and max in order not to go outside the grid
                next_y = min(max(y + dy, 0), grid_size_y - 1) # min and max in order not to go outside the grid
                
                next_s = (next_x, next_y)
                if next_s == goal:
                    r = 1 
                else:
                    r = -1
                P[s].append((1.0/len(A), next_s, r))
    
    # Value function calculation (Iterative policy evaluation)
    n_iter = 0
    delta = 0
    while delta >= THETA or n_iter == 0:   
        delta = 0
        for s in S:
            v = V[s]
            V[s] = sum(prob * (r + GAMMA * V[s_prime]) for prob, s_prime, r in P[s])
            delta = max(delta, abs(v - V[s]))
        n_iter += 1
    
    # Value function visualization
    grid_values = np.zeros((grid_size_x, grid_size_y))
    for (x, y), value in V.items():
        grid_values[x, y] = value
    
    print('Final V:')
    print(grid_values)
