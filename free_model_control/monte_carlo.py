# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 17:35:53 2025

@author: andres.sanchez
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
    
def MC_line(line_size,  n_episodes = 1000, n_steps = 20):

    # --- CONFIGURACIÓN DEL PROBLEMA ---
    goal = line_size
    # goal = 8
    start = 0
    
    S = [m for m in range(line_size +1)]
    A = [-1, 1]  # izquierda, quieto, derecha
    Q = {(s, a): 0.0 for s in S for a in A}
    N = {(s, a): 0 for s in S for a in A}  # contador de visitas
    pi = {s: np.random.choice(A) for s in S}
    
    GAMMA = 0.9
    EPSILON = 0.2  # exploración
    
    for ep in range(n_episodes):
        if ep % 100 == 0:
            print(f'Episode: {ep}')
        s = start
        episode = []
    
        # --- GENERAR EPISODIO CON ε-GREEDY ---
        for st in range(n_steps):
            if np.random.rand() < EPSILON:
                a = np.random.choice(A)
            else:
                a = max(A, key=lambda a_: Q[s, a_])
            
            s_prima = max(0, min(s + a, goal))
            r = 1 if s_prima == goal else -1
            episode.append((s, a, r))
            s = s_prima
            if s == goal:
                break
    
        # --- MONTE CARLO PRIMERA VISITA ---
        visited = set()
        G = 0
        for s, a, r in reversed(episode):
            G = GAMMA * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                N[(s, a)] += 1  # incrementar contador de visitas
                alpha = 1.0 / N[(s, a)]  # paso adaptativo (media incremental)
                Q[(s, a)] += alpha * (G - Q[(s, a)])
    
        # --- MEJORAR POLÍTICA ---
        for s in S:
            a_star = max(A, key=lambda a: Q[s, a])
            probs = []
            for a in A:
                if a == a_star:
                    probs.append(1 - EPSILON + EPSILON/len(A))
                else:
                    probs.append(EPSILON/len(A))
            pi[s] = np.random.choice(A, p=probs)
    
    print("\nQ-values finales:")
    for s in S:
        print(f"Estado {s}: ", {a: round(Q[s, a], 2) for a in A})
    

def MC_grid(grid_size_x, grid_size_y,
             N_EPISODES = 1000,
             N_STEPS = 10, goal = (3,3),
             start = (0,0)):
    
    def print_policy(pi, grid_size_x, grid_size_y, start, goal):
        for i in range(grid_size_x):
            row = ''
            for j in range(grid_size_y):
                s = (i, j)
                if s == start:
                    row += 'S '
                elif s == goal:
                    row += 'G '
                else:
                    a = pi.get(s, None)
                    if a == (0, 1): row += '→ '
                    elif a == (1, 0): row += '↓ '
                    elif a == (-1, 0): row += '↑ '
                    elif a == (0, -1): row += '← '
                    else: row += '* '
            print(row)
    
    S = [(m,n) for m in range(grid_size_x) for n in range(grid_size_y)]
    A = [(0,1), (1,0), (-1,0), (0,-1)]  
    Q = {(s, a): 0.0 for s in S for a in A}
    N = {(s, a): 0 for s in S for a in A}  # contador de visitas
    pi = {s: random.choice(A) for s in S}
    
    GAMMA = 0.9
    EPSILON = 0.1
    
    for ep in range(N_EPISODES):
        if ep % 100 == 0:
            print(f'Episode: {ep}')
        EPSILON = max(0.01, 0.2 / (1 + ep / 10)) # A MEDIDA QUE HACEMOS MAS EPISODIOS EXPLORAMOS MENOS
        episodes = []
        s = start
        for st in range(N_STEPS):
            if random.random() < EPSILON:
                a = random.choice(A)
            else:
                a = pi[s]
                
            next_x = min(max(s[0] + a[0], 0), grid_size_x - 1) # min and max in order not to go outside the grid
            next_y = min(max(s[1] + a[1], 0), grid_size_y - 1) # min and max in order not to go outside the grid
            s_prima = (next_x, next_y)
           
            if s_prima == goal:
                r = 10
            else:
                r = -1
            
            episodes.append((s_prima, r, a))
            
            s = s_prima
            if s_prima == goal:
                break
        
        G = 0
        visited = set()
        for s, r, a in reversed(episodes):
            G = GAMMA * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                N[(s, a)] += 1  # incrementar contador de visitas
                alpha = 1.0 / N[(s, a)]  # paso adaptativo (media incremental)
                Q[(s, a)] += alpha * (G - Q[(s, a)])
        
        for s in S:
            a_star = max(A, key=lambda a: Q[s, a])
            probs = []
            for a in A:
                if a == a_star:
                    probs.append(1 - EPSILON + EPSILON / len(A))
                else:
                    probs.append(EPSILON / len(A))
            pi[s] = A[np.random.choice(list(range(len(A))), p=probs)]
            
    # print("\nLearned Policy (S = start, G = goal):")
    # print_policy(pi, grid_size_x, grid_size_y, start, goal)
    arrow_map = {(0,1): '→', (1,0): '↓', (-1,0): '↑', (0,-1): '←'}
    print("\nMejor acción por estado:")
    for x in range(grid_size_x):  # rows
        row = ""
        for y in range(grid_size_y):  # columns
            state = (x, y)
            if state == goal:
                row += " G "
            else:
                best_a = max(A, key=lambda a_: Q[(state, a_)])
                row += f" {arrow_map[best_a]} "
        print(row)
     
