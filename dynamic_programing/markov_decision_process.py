# -*- coding: utf-8 -*-
"""
Created on 2025

@author: andres.sanchez
"""

from collections import defaultdict
import numpy as np 
import math

def markov_decision_process_line(line_size): 
    goal = line_size 
    S = [i for i in range(line_size + 1)]
    A = [-1, 0, 1]
    pi = {s:0 for s in S}
    V = {s:0 for s in S}
    
    THETA = 0.01
    GAMMA = 0.9
    
    P = defaultdict(list)
    for s in S:
        if s == goal:
            for a in A:
                P[s, a].append((1, s, 0)) # if we set reward to 0, due to precision we can not properly see the V grid
        else:
            for a in A:
               
                new_s = max(0, min(line_size, s + a))
                
                if new_s == goal:
                    r = 0
                else:
                    r = -1
                
                P[s, a].append((1/len(A), new_s, r))
    
    def improve_value():
        delta = math.inf
        while delta > THETA:
            delta = 0
            for s in S:
                v_old = V[s]
                a = pi[s]
                v = 0
                for prob, s_prima, r in P[s, a]:
                    v += prob * (GAMMA * V[s_prima] + r)
                V[s] = v
                delta = max(delta, abs(v_old - v))
        return delta
    
    def improve_policy():
        policy_stable = True
        
        for s in S:
            v_max = -math.inf
            old_a = pi[s]
            for a in A:
                v = 0
                for prob, s_prima, r in P[s,a]:
                    v += prob * (GAMMA* V[s_prima] + r)
                if v > v_max:
                    v_max = v
                    pi[s] = a
            
            if pi[s] != old_a:
                policy_stable = False
        
        return policy_stable
        
    policy_stable = False
    n_iterations = 0
    while not policy_stable or n_iterations == 0:
       delta = improve_value()
       policy_stable = improve_policy()
       print(f'V difference: {delta}, iteration: {n_iterations}')
       n_iterations += 1         
         
    grid_values = np.zeros(line_size+1)
    for x, value in V.items():
        grid_values[x] = value         
     
    print('Final V:')
    print(grid_values)             


def markov_decision_process_gridw(grid_size_x, 
                          grid_size_y, 
                          goal):
    # mdp con la misma probabilidad de avanzar en todas las direcciones
    # misma probabilidad en matriz de transicion
    THETA = 0.01
    GAMMA = 0.9
    
    A = [(0,1), (1,0), (-1,0), (0,-1)]
    S = [(n,m) for n in range(grid_size_x) for m in range(grid_size_y)]
    V = {s:0 for s in S}
    pi = {s:(0,0) for s in S}
    
    # FIRST WE SIMULATE REALITY: ANY ACTION THAT CAN HAPPEN IN ANY GIVEN STATE
    P = defaultdict(list)
    for s in S:
        if s == goal:
            for a in A:
                P[s, a] = [(1.0, s, 0)] # value of the goal position will always be 0 because there will be no reward and as the initial value is 0, it will never be updated
        else:
            for a in A:
                x, y = s
                dx, dy = a
                
                next_x = min(max(x + dx, 0), grid_size_x - 1) # min and max in order not to go outside the grid
                next_y = min(max(y + dy, 0), grid_size_y - 1) # min and max in order not to go outside the grid
                
                next_s = (next_x, next_y)
        
                if next_s == goal:
                    r = 10 
                else:
                    r = -1
                P[(s, a)].append((1.0/len(A), next_s, r))
    
    # THEN WE ADJUST THE VALUE FUNCTION GIVEN A POLICY FUNCTION
    def evaluate_how_good_policy():
        n_iter = 0
        delta = 0
        while delta >= THETA or n_iter == 0:
            delta = 0
            for s in S:
                a = pi[s]
                v_old = V[s]
                
                v_mean = 0
                for prob, s_prime, r in P[s, a]:
                    v_mean += prob * (r + GAMMA * V[s_prime]) # here the value function is updated
                V[s] = v_mean
                
                delta = max(delta, abs(v_old - V[s]))
            n_iter += 1
        
        return delta 
    
    # WE ADJUST THE POLICY FUNCTION GIVEN THE RESULTING VALUE FUNCTION
    def improve_policy():
        policy_stable = True
        for s in S:
            old_action = pi[s]
            max_v = -math.inf
            for a in A:
                v = 0
                for my_tuple in P[s, a]:
                    prob, s_prime, r = my_tuple
                    v += prob * (r + GAMMA * V[s_prime])
                
                if v >= max_v:
                    max_v = v
                    pi[s] = a
                    
            if old_action != pi[s]:
                policy_stable = False
        
        return policy_stable
    
    n_policy_iterations = 0
    delta = 0
    policy_stable = False
    while policy_stable == False:
        delta = evaluate_how_good_policy()
        policy_stable = improve_policy()
        print("# Policy Iterations:", n_policy_iterations, f'delta: {delta}')
        n_policy_iterations += 1
    
    grid_values = np.zeros((grid_size_x, grid_size_y))
    for (x, y), value in V.items():
        grid_values[x, y] = value
    
    print('Final V:')
    print(grid_values)
    print(f'The goal value {goal} is never updated so it must be 0')
    

def mdp_gridw_randact(grid_size_x, 
                      grid_size_y, 
                      goal,
                      frozen_lake = [['S', 'F', 'F', 'H'],
                                      ['F', 'H', 'F', 'F'],
                                      ['F', 'F', 'H', 'F'],
                                      ['H', 'F', 'F', 'G']
                                    ]):
    
    A = [(0,1), (1,0), (-1,0), (0,-1)]
    S = [(n,m) for n in range(grid_size_x) for m in range(grid_size_y)]
    V = {s: 0 for s in S}
    pi = {s:(0,0) for s in S}
    GAMMA = 0.9
    
    frozen_lake = [j for i in frozen_lake for j in i]
    frozen_lake = dict(zip(S, frozen_lake))
    
    # Se introduce la posibilidad de desviarse 10% hacia las dos direcciones perpendiculares al movimiento elegido
    perpendicular = {
        (0,1): [(-1,0), (1,0)],   # Si eliges ir a la derecha, puedes desviarte hacia arriba o abajo
        (1,0): [(0,1), (0,-1)],   # Si eliges ir abajo, puedes desviarte hacia derecha o izquierda
        (-1,0): [(0,-1), (0,1)],  # Si eliges ir arriba, puedes desviarte hacia izquierda o derecha
        (0,-1): [(1,0), (-1,0)]   # Si eliges ir a la izquierda, puedes desviarte hacia abajo o arriba
    }
    
    P = defaultdict(list)
    for position, floor_type in frozen_lake.items():
        if frozen_lake[position] == 'H':
            for a in A:
                P[(position, a)].append((1, position, -10))
        elif frozen_lake[position] == 'G':
            for a in A:
                P[(position, a)].append((1, position, 0))
        else:    
            for a in A:
                x, y = position
                dx, dy = a
                
                next_x = min(max(x + dx, 0), grid_size_x - 1) # min and max in order not to go outside the grid
                next_y = min(max(y + dy, 0), grid_size_y - 1) # min and max in order not to go outside the grid
                next_s = (next_x, next_y)
                
                if frozen_lake[next_s] == 'G':
                    r = 0
                elif frozen_lake[next_s] == 'H':
                    r = -5
                else:
                    r = -1
                    
                P[(position, a)].append((0.8, next_s, r))
       
                # perpendiculares
                left, right = perpendicular[a]
                for dx_p, dy_p in [left, right]:
                    next_x = min(max(x + dx_p, 0), grid_size_x - 1)
                    next_y = min(max(y + dy_p, 0), grid_size_y - 1)
                    next_s = (next_x, next_y)
                    P[(position, a)].append((0.1, next_s, r))
    
    # AQUI SE AJUSTA LA FUNCION V
    def evaluate_value_function():
        delta = 0
        THETA = 0.01
        n_iter = 0
        while delta >= THETA or n_iter == 0:  
            print(n_iter)
            delta = 0
            for s in S:
                old_v = V[s]
                a = pi[s]
                v = 0
                for prob, s_prima, r in P[(s,a)]:
                    v += prob * (r + GAMMA * V[s_prima])
                V[s] = v
                delta = max(delta, abs(v-old_v))
            n_iter += 1
        return delta
    
    # AQUI SE AJUSTA LA FUNCION PI POLICY
    def improve_policy(): 
        GAMMA = 0.9
        policy_stable = True
        for s in S:
            max_v = -math.inf
            old_action = pi[s]
            for a in A:
                v = 0
                for prob, s_prima, r in P[(s, a)]:
                    v += prob * (GAMMA * V[s_prima] + r)
                if v > max_v:
                    max_v = v
                    pi[s] = a
                   
            if old_action != pi[s]:
                policy_stable = False
        
        return policy_stable
    
    policy_stable = False
    n_iterations = 0
    while not policy_stable: 
        delta = evaluate_value_function() 
        policy_stable = improve_policy()
        n_iterations += 1
        print(n_iterations, delta)
        
    grid_values = np.zeros((grid_size_x, grid_size_y))
    for (x, y), value in V.items():
        grid_values[x, y] = value         
    
    print('Final V:')     
    print(grid_values)           
