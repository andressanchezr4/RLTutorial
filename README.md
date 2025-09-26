# RLTutorial

**Learn Reinforcement Learning through hands-on problems: pushing a cart along a line and walking on a grid.**

---

## 📂 Repository Structure

### **1. dynamic_programing**

- **policy_iteration.py**
  - **Equation:**  
    $$
    V(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s'|s, a) V(s') \right]
    $$
  - Policy Evaluation and Improvement.
  - Bellman Equations.
  - Convergence Criteria.
  - Optimal Policy Extraction.
  - Computational Complexity Analysis.

- **value_iteration.py**
  - **Equation:**  
    \[
    V(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s'|s, a) V(s') \right]
    \]
  - Iterative Value Updates.
  - Bellman Optimality Equations.
  - Convergence to Optimal Value Function.
  - Policy Extraction from Value Function.
  - Computational Complexity Considerations.

---

### **2. free_model_control**

- **q_learning.py**
  - **Equation:**  
    \[
    Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right]
    \]
  - Off-policy Learning.
  - Temporal Difference (TD) Learning.
  - Exploration vs. Exploitation.
  - Q-value Updates.
  - Policy Improvement Strategies.

- **sarsa.py**
  - **Equation:**  
    \[
    Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]
    \]
  - On-policy Learning.
  - Temporal Difference (TD) Learning.
  - Exploration vs. Exploitation.
  - Q-value Updates.
  - Policy Improvement Strategies.

---

### **3. free_model_prediction**

- **mc_prediction.py**
  - **Equation:**  
    \[
    G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k
    \]
  - Return Estimation.
  - Bootstrapping.
  - First-visit vs. Every-visit MC.
  - Learning Rate Schedules.
  - Bias-Variance Tradeoff.

- **td_prediction.py**
  - **Equation:**  
    \[
    V(s_t) \leftarrow V(s_t) + \alpha \left[ r_{t+1} + \gamma V(s_{t+1}) - V(s_t) \right]
    \]
  - Temporal Difference (TD) Learning.
  - Bootstrapping.
  - Learning Rate Schedules.
  - Bias-Variance Tradeoff.
  - Convergence Analysis.

---

### **4. value_function_approx_control**

- **linear_q_learning.py**
  - **Equation:**  
    \[
    Q(s, a) \approx \theta^T \phi(s, a)
    \]
  - Linear Function Approximation.
  - Feature Engineering.
  - Gradient Descent Optimization.
  - Overfitting Prevention.
  - Experience Replay.

- **neural_network_q.py**
  - **Equation:**  
    \[
    Q(s, a) \approx f_{\theta}(s, a)
    \]
  - Neural Network Function Approximation.
  - Deep Q-Networks (DQN).
  - Experience Replay.
  - Target Networks.
  - Loss Function Design.

---

### **5. value_function_approx_prediction**

- **linear_value_prediction.py**
  - **Equation:**  
    \[
    V(s) \approx \theta^T \phi(s)
    \]
  - Linear Function Approximation.
  - Feature Engineering.
  - Gradient Descent Optimization.
  - Overfitting Prevention.
  - Convergence Analysis.

- **neural_network_value.py**
  - **Equation:**  
    \[
    V(s) \approx f_{\theta}(s)
    \]
  - Neural Network Function Approximation.
  - Deep Value Networks.
  - Loss Function Design.
  - Regularization Techniques.
  - Hyperparameter Tuning.
