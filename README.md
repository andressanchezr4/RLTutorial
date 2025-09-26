# RLTutorial

Hands-on Reinforcement Learning with problems:
-  Pushing a cart along a line.
-  Grid World.

All the methods in this repository are **implementations or approximations of the Bellman equation**. 

The Bellman equation expresses (1) the value of a state as the expected return from taking an action and (2) following a policy thereafter. 

<p align="center">
$` V(s) = max_a [ R(s,a) + γ Σ_{s'} P(s'|s,a) V(s') ] `$
</p>


### 1. **Dynamic Programming**

- **policy_iteration.py**
  - **Equation:**  
$` V(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [ R(s,a,s') + \gamma V(s') ] `$
  - Policy Evaluation and Improvement.
  - Bellman Equations.
  - Convergence Criteria.
  - Optimal Policy Extraction.
  - Computational Complexity Analysis.

- **value_iteration.py**
  - **Equation:**  
$` V(s) = \max_a \sum_{s'} P(s'|s,a) [ R(s,a,s') + \gamma V(s') ] `$
  - Iterative Value Updates.
  - Bellman Optimality Equations.
  - Convergence to Optimal Value Function.
  - Policy Extraction from Value Function.
  - Computational Complexity Considerations.

---

### 2. **Free Model Prediction**

- **mc_prediction.py**
  - **Equation:**  
$` G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k `$
  - Return Estimation.
  - Bootstrapping.
  - First-visit vs. Every-visit MC.
  - Learning Rate Schedules.
  - Bias-Variance Tradeoff.

- **td_prediction.py**
  - **Equation:**  
$` V(s_t) <- V(s_t) + \alpha [ r_{t+1} + \gamma V(s_{t+1}) - V(s_t) ] `$
  - Temporal Difference (TD) Learning.
  - Bootstrapping.
  - Learning Rate Schedules.
  - Bias-Variance Tradeoff.
  - Convergence Analysis.

---

### 3. **Free Model Control**

- **q_learning.py**
  - **Equation:**  
$` Q(s_t, a_t) <- Q(s_t, a_t) + \alpha [ r_{t+1} + \gamma max_a Q(s_{t+1}, a) - Q(s_t, a_t) ] `$
  - Off-policy Learning.
  - Temporal Difference (TD) Learning.
  - Exploration vs. Exploitation.
  - Q-value Updates.
  - Policy Improvement Strategies.

- **sarsa.py**
  - **Equation:**  
$` Q(s_t, a_t) <- Q(s_t, a_t) + \alpha [ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) ] `$
  - On-policy Learning.
  - Temporal Difference (TD) Learning.
  - Exploration vs. Exploitation.
  - Q-value Updates.
  - Policy Improvement Strategies.

---

### 4. **Value Function Approximation Prediction**

- **linear_value_prediction.py**
  - **Equation:**  
$` V(s) ≈ θ^T φ(s) `$
  - Linear Function Approximation.
  - Feature Engineering.
  - Gradient Descent Optimization.
  - Overfitting Prevention.
  - Convergence Analysis.

- **neural_network_value.py**
  - **Equation:**  
$` V(s) ≈ f_θ(s) `$
  - Neural Network Function Approximation.
  - Deep Value Networks.
  - Loss Function Design.
  - Regularization Techniques.
  - Hyperparameter Tuning.

---

### 5. **Value Function Approximation Control**

- **linear_q_learning.py**
  - **Equation:**  
$` Q(s,a) ≈ θ^T φ(s,a) `$
  - Linear Function Approximation.
  - Feature Engineering.
  - Gradient Descent Optimization.
  - Overfitting Prevention.
  - Experience Replay.

- **neural_network_q.py**
  - **Equation:**  
$` Q(s,a) ≈ f_θ(s,a) `$
  - Neural Network Function Approximation.
  - Deep Q-Networks (DQN).
  - Experience Replay.
  - Target Networks.
  - Loss Function Design.


