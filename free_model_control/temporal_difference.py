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

   
def TD0_line(line_size, n_episodes=1000, n_steps=20):
    # --- CONFIGURACIÓN DEL PROBLEMA ---
    goal = line_size
    start = 0

    S = [m for m in range(line_size + 1)]
    A = [-1, 1]  # izquierda, derecha
    Q = {(s, a): 0.0 for s in S for a in A}

    GAMMA = 0.9
    EPSILON = 0.2  # exploración
    ALPHA = 0.1

    for ep in range(n_episodes):
        if ep % 100 == 0:
            print(f"Episode: {ep}")

        s = start
        for st in range(n_steps):
            if np.random.rand() < EPSILON:
                a = np.random.choice(A)
            else:
                a = max(A, key=lambda a_: Q[(s, a_)])

            s_prima = max(0, min(s + a, goal))
            if s_prima == goal:
                r = 1 
            else:
                r = -1

            # --- ACTUALIZACIÓN TD(0) CONTROL (Q-learning) ---
            best_next_action = max(A, key=lambda a_: Q[(s_prima, a_)])
            Q[(s, a)] += ALPHA * (r + GAMMA * Q[(s_prima, best_next_action)] - Q[(s, a)])

            s = s_prima
            if s == goal:
                break

    print("\nQ-values finales:")
    for s in S:
        print(f"Estado {s}: ", {a: round(Q[(s, a)], 2) for a in A})
    
    return Q

def TD0_gridw(grid_size_x, grid_size_y,
             N_EPISODES = 10000,
             N_STEPS = 10, goal = (3,3),
             start = (0,0)):

    S = [(m,n) for m in range(grid_size_x) for n in range(grid_size_y)]
    A = [(0,1), (1,0), (-1,0), (0,-1)]  
    Q = {(s, a): 0.0 for s in S for a in A}
    
    GAMMA = 0.9
    EPSILON = 0.2
    ALPHA = 0.1
    s = start
    for st in range(N_STEPS):
        if np.random.rand() < EPSILON:
            a = A[np.random.choice(len(A))]
        else:
            a = None
            v_max = float('-inf')
            for act in A:
                if Q[(s, act)] > v_max:
                    v_max = Q[(s, act)]
                    a = act
    
        # --- next state ---
        s_prima_x = min(max(s[0]+a[0], 0), grid_size_x-1)
        s_prima_y = min(max(s[1]+a[1], 0), grid_size_y-1)
        s_prima = (s_prima_x, s_prima_y)
        if s_prima == goal:
            r = 10 
        else:
            r = -1
    
        best_next_action = None
        v_max = float('-inf')
        for act in A:
            if Q[(s_prima, act)] > v_max:
                v_max = Q[(s_prima, act)]
                best_next_action = act
    
        Q[(s, a)] += ALPHA * (r + GAMMA * Q[(s_prima, best_next_action)] - Q[(s, a)])
    
        s = s_prima
        if s == goal:
            break
    
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

def TD0_nstep_gridw(grid_size_x=4, grid_size_y=4,
                      N_EPISODES=10000,
                      N_STEPS=10,
                      N_STEP=3,
                      goal=(3,3),
                      start=(0,0)):

    # --- SETUP ---
    S = [(x,y) for x in range(grid_size_x) for y in range(grid_size_y)]
    A = [(0,1), (1,0), (-1,0), (0,-1)]  # right, down, left, up
    Q = {(s,a): 0.0 for s in S for a in A}

    GAMMA = 0.9
    EPSILON = 0.2
    ALPHA = 0.1

    # --- TRAINING ---
    for ep in range(N_EPISODES):
        if ep % 1000 == 0:
            print(f"Episode: {ep}")

        s = start
        # choose first action epsilon-greedy
        if np.random.rand() < EPSILON:
            a = A[np.random.choice(len(A))]
        else:
            a = max(A, key=lambda act: Q[(s, act)])

        buffer = deque()

        for t in range(N_STEPS):
            # take action
            s_next = (min(max(s[0]+a[0],0),grid_size_x-1),
                      min(max(s[1]+a[1],0),grid_size_y-1))
            r = 10 if s_next == goal else -1

            buffer.append((s, a, r))

            # choose next action epsilon-greedy
            if np.random.rand() < EPSILON:
                a_next = A[np.random.choice(len(A))]
            else:
                a_next = max(A, key=lambda act: Q[(s_next, act)])

            # update Q if buffer has enough steps
            if len(buffer) >= N_STEP:
                G = 0.0
                for idx, (_, _, r_i) in enumerate(buffer):
                    G += (GAMMA**idx) * r_i
                s_n, a_n, _ = buffer.popleft()  # oldest step
                G += (GAMMA**N_STEP) * Q[(s_next, a_next)]  # bootstrapped value
                Q[(s_n, a_n)] += ALPHA * (G - Q[(s_n, a_n)])

            s, a = s_next, a_next

            if s == goal:
                # flush the remaining steps in buffer
                while buffer:
                    G = 0.0
                    for idx, (_, _, r_i) in enumerate(buffer):
                        G += (GAMMA**idx) * r_i
                    s_n, a_n, _ = buffer.popleft()
                    Q[(s_n, a_n)] += ALPHA * (G - Q[(s_n, a_n)])
                break

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

def TD0_lambda_gridw(grid_size_x=4, grid_size_y=4,
                   N_EPISODES=10000,
                   N_STEPS=20,
                   LAMBDA=0.8,
                   goal=(3,3),
                   start=(0,0)):

    # --- SETUP ---
    S = [(x,y) for x in range(grid_size_x) for y in range(grid_size_y)]
    A = [(0,1), (1,0), (-1,0), (0,-1)]  # right, down, left, up
    Q = {(s,a): 0.0 for s in S for a in A}

    GAMMA = 0.9
    EPSILON = 0.2
    ALPHA = 0.1

    for ep in range(N_EPISODES):
        if ep % 1000 == 0:
            print(f"Episode: {ep}")

        s = start
        # choose first action epsilon-greedy
        if np.random.rand() < EPSILON:
            a = A[np.random.choice(len(A))]
        else:
            a = max(A, key=lambda act: Q[(s, act)])

        # eligibility traces
        # el eligibility trace hace que si un paso se toma muchas veces, 
        # su impacto en el valor esperado de una accion en un estado poco a 
        # poco sea mas pequeño
        E = {(s_, a_): 0.0 for s_ in S for a_ in A}

        for t in range(N_STEPS):
            # take action
            s_next = (min(max(s[0]+a[0],0), grid_size_x-1),
                      min(max(s[1]+a[1],0), grid_size_y-1))
            
            if s_next == goal:
                r = 10 
            else:
                r = -1

            if np.random.rand() < EPSILON:
                a_next = A[np.random.choice(len(A))]
            else:
                a_next = max(A, key=lambda act: Q[(s_next, act)])

            # TD error
            delta = r + GAMMA * Q[(s_next, a_next)] - Q[(s, a)]

            # increment eligibility trace for current state-action
            E[(s, a)] += 1

            # update all Q-values
            for s_, a_ in Q.keys():
                Q[(s_, a_)] += ALPHA * delta * E[(s_, a_)]
                E[(s_, a_)] *= GAMMA * LAMBDA  # decay eligibility

            s, a = s_next, a_next

            if s == goal:
                break
    
    # --- PRINT Q-values ---
    print("\nQ-values finales:")
    for s in S:
        print(f"Estado {s}: ", {a: round(Q[(s, a)], 2) for a in A})
    
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

