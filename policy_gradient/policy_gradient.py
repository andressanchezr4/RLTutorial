# -*- coding: utf-8 -*-
"""
Created on 2025

@author: andres.sanchez
"""

import numpy as np

class GridWorld:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.start_state = (0, 0)
        self.goal_state = (grid_size - 1, grid_size - 1)
        self.reset()

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # left
            y = max(0, y - 1)
        elif action == 3:  # right
            y = min(self.grid_size - 1, y + 1)
        self.state = (x, y)
        
        if self.state == self.goal_state:
            reward = 1.0 
        else:
            reward = 0.0 
        done = self.state == self.goal_state
        return self.state, reward, done

class SoftmaxPolicy:
    def __init__(self, n_states, n_actions):
        self.theta = np.random.randn(n_states, n_actions) * 0.01
        self.n_states = n_states
        self.n_actions = n_actions

    def get_probs(self, state_idx):
        logits = self.theta[state_idx]
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def sample_action(self, state_idx):
        probs = self.get_probs(state_idx)
        return np.random.choice(self.n_actions, p=probs)


# ----- Utility Functions -----
def state_to_idx(state, grid_size):
    return state[0] * grid_size + state[1]


def run_episode(env, policy, max_steps=50):
    state = env.reset()
    total_reward = 0
    for _ in range(max_steps):
        s_idx = state_to_idx(state, env.grid_size)
        a = policy.sample_action(s_idx)
        state, reward, done = env.step(a)
        total_reward += reward
        if done:
            break
    return total_reward


def finite_difference_gradient(env, policy, episodes_per_eval=5, delta=1e-2):
    grad = np.zeros_like(policy.theta)
    base_reward = np.mean([run_episode(env, policy) for _ in range(episodes_per_eval)])
    for i in range(policy.theta.shape[0]):
        for j in range(policy.theta.shape[1]):
            old_val = policy.theta[i, j]

            # θ+δ
            policy.theta[i, j] = old_val + delta
            reward_plus = np.mean([run_episode(env, policy) for _ in range(episodes_per_eval)])

            # θ-δ
            policy.theta[i, j] = old_val - delta
            reward_minus = np.mean([run_episode(env, policy) for _ in range(episodes_per_eval)])

            grad[i, j] = (reward_plus - reward_minus) / (2 * delta)

            policy.theta[i, j] = old_val  # restore

    return grad, base_reward


# ----- Training Loop -----
grid_size = 5
n_states = grid_size * grid_size
n_actions = 4
env = GridWorld(grid_size)
policy = SoftmaxPolicy(n_states, n_actions)

alpha = 1e-1  # learning rate
n_iters = 50

for it in range(n_iters):
    grad, J = finite_difference_gradient(env, policy)
    policy.theta += alpha * grad
    print(f"Iter {it+1}, Expected Return: {J:.3f}")

##############################################################

# --- Reutilizamos GridWorld y SoftmaxPolicy del ejemplo anterior ---

def run_episode_with_trajectory(env, policy, max_steps=50, gamma=1.0):
    """
    Devuelve (states, actions, rewards) de un episodio completo.
    """
    state = env.reset()
    states, actions, rewards = [], [], []
    for _ in range(max_steps):
        s_idx = state_to_idx(state, env.grid_size)
        a = policy.sample_action(s_idx)
        next_state, reward, done = env.step(a)
        states.append(s_idx)
        actions.append(a)
        rewards.append(reward)
        state = next_state
        if done:
            break
    return states, actions, rewards


def compute_returns(rewards, gamma=1.0):
    """
    Calcula G_t = r_t + γr_{t+1} + ... para cada paso t.
    """
    G = np.zeros(len(rewards))
    running_sum = 0
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma * running_sum
        G[t] = running_sum
    return G


def reinforce(env, policy, alpha=0.1, gamma=1.0, episodes=2000):
    for ep in range(episodes):
        states, actions, rewards = run_episode_with_trajectory(env, policy)
        returns = compute_returns(rewards, gamma)

        # Gradiente Monte Carlo
        grad_theta = np.zeros_like(policy.theta)
        for s, a, G in zip(states, actions, returns):
            probs = policy.get_probs(s)
            # grad log π(a|s) = (one_hot(a) - probs)
            one_hot = np.zeros_like(probs)
            one_hot[a] = 1.0
            grad_theta[s] += (one_hot - probs) * G

        policy.theta += alpha * grad_theta  # actualización de parámetros

        if (ep + 1) % 100 == 0:
            total_reward = np.sum(rewards)
            print(f"Episode {ep+1}, Return: {total_reward:.2f}")


# --- Entrenamiento ---
grid_size = 5
n_states = grid_size * grid_size
n_actions = 4
env = GridWorld(grid_size)
policy = SoftmaxPolicy(n_states, n_actions)

reinforce(env, policy, alpha=1e-2, gamma=1.0, episodes=1000)


################################################################
# --- Reutilizamos GridWorld y SoftmaxPolicy del ejemplo anterior ---

class ValueFunction:
    def __init__(self, n_states):
        self.V = np.zeros(n_states)

    def predict(self, s):
        return self.V[s]

    def update(self, s, target, alpha_v):
        self.V[s] += alpha_v * (target - self.V[s])


def actor_critic(env, policy, alpha_theta=0.1, alpha_v=0.1, gamma=0.99, episodes=2000, max_steps=50):
    critic = ValueFunction(policy.n_states)

    for ep in range(episodes):
        state = env.reset()
        s_idx = state_to_idx(state, env.grid_size)
        total_reward = 0

        for t in range(max_steps):
            probs = policy.get_probs(s_idx)
            a = np.random.choice(policy.n_actions, p=probs)
            next_state, r, done = env.step(a)
            s_next_idx = state_to_idx(next_state, env.grid_size)

            # --- Critic update ---
            td_target = r + gamma * critic.predict(s_next_idx)
            td_error = td_target - critic.predict(s_idx)
            critic.update(s_idx, td_target, alpha_v)

            # --- Actor update ---
            one_hot = np.zeros_like(probs)
            one_hot[a] = 1.0
            grad_log_pi = one_hot - probs
            policy.theta[s_idx] += alpha_theta * grad_log_pi * td_error

            total_reward += r
            s_idx = s_next_idx

            if done:
                break

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}, Return: {total_reward:.2f}")


# --- Entrenamiento ---
grid_size = 5
n_states = grid_size * grid_size
n_actions = 4
env = GridWorld(grid_size)
policy = SoftmaxPolicy(n_states, n_actions)

actor_critic(env, policy, alpha_theta=1e-2, alpha_v=1e-1, gamma=0.99, episodes=1000)
