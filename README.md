## Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

### Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

### Problem Statement
To implement the Q-Learning control algorithm using the Gymnasium FrozenLake-v1 environment and enable the agent to learn an optimal policy for reaching the goal state while avoiding the holes.

### Software Requirements
1. Python 3.x
2. Google Colab / Jupyter Notebook
3. Gymnasium
4. NumPy
5. Matplotlib


### Environment Description
FrozenLake-v1 is a grid-world environment consisting of 16 states arranged in a 4×4 grid. The agent starts from the initial state and must reach the goal while avoiding holes.

The environment has 4 possible actions:
<ul><li>0 – Left (L)</li>
<li>1 – Down (D)</li>
<li>2 – Right (R)</li>
<li>3 – Up (U)</li></ul>

The agent receives a reward of 1 for reaching the goal and 0 for other transitions. The Q-Learning algorithm learns the best action for each state through repeated interaction with the environment.


### Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

### Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

### Algorithm
1. Create the FrozenLake-v1 environment.
2. Initialize the Q-table with zeros.
3. Set learning rate, discount factor, and epsilon values.
4. Reset the environment and obtain the initial state.
5. Select an action using epsilon-greedy.
6. Perform the action and observe the reward and next state.
7. Update the Q-value using the Q-Learning update rule.
8. Repeat until the episode ends.
9. Reduce epsilon gradually for less exploration.
10. Repeat for the required number of episodes.
11. Extract the state-value function and learned policy from the Q-table.

### Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=False)

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
num_episodes = 20000
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------
Q = np.zeros((env.observation_space.n, env.action_space.n))
episode_rewards = []

# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------
def choose_action(state):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------
for episode in range(num_episodes):

    state, info = env.reset()
    total_reward = 0
    done = False

    while not done:

        action = choose_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# -------------------------------------------------
# Display Functions
# -------------------------------------------------
def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))

def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)

# -------------------------------------------------
# Output
# -------------------------------------------------
state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)

print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(learned_policy)

average_reward = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", average_reward)

# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------
window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Q-Learning Curve - FrozenLake")
plt.grid(True)
plt.show()

env.close()
```

### Output
```
For,
num_episodes = 20000
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995
```
#### Final Q-table:
<img width="137" height="192" alt="image" src="https://github.com/user-attachments/assets/065ebf81-91a7-4124-855c-97eefac8fdad" />

#### Estimated State-Value Function:
<img width="166" height="62" alt="image" src="https://github.com/user-attachments/assets/dbffcccc-e292-4e63-9874-e5931de59435" />

#### Learned Policy:
<img width="115" height="67" alt="image" src="https://github.com/user-attachments/assets/b8d2d074-1ac3-4f10-9a6d-416877ac5a01" />

#### Average reward over last 1000 episodes: 
<img width="472" height="307" alt="image" src="https://github.com/user-attachments/assets/b0e8482f-e00d-417c-b581-6df9dafb8437" />

```
For,
num_episodes = 10000      
alpha = 0.5               
gamma = 0.8               
epsilon = 0.8             
epsilon_min = 0.1         
epsilon_decay = 0.995
```
#### Final Q-table:
<img width="252" height="320" alt="image" src="https://github.com/user-attachments/assets/d8d8fb08-bdf3-4033-8308-2881156d8de3" />

#### Estimated State-Value Function:
<img width="263" height="97" alt="image" src="https://github.com/user-attachments/assets/114f193e-0173-4325-b1cd-d0b4fe10d72e" />

#### Learned Policy:
<img width="165" height="100" alt="image" src="https://github.com/user-attachments/assets/0e10a735-e0ac-45ab-94fd-df0e4991a2f4" />

#### Average reward over last 1000 episodes: 
<img width="756" height="518" alt="image" src="https://github.com/user-attachments/assets/f4bb0247-ebe1-4057-a56b-5a152d61e2a0" />


### Result
The Q-Learning control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The agent learned Q-values and obtained a policy for reaching the goal while avoiding holes.

### Inference
The Q-Learning agent gradually learns the optimal policy through exploration and repeated training. Changing the hyperparameters affects the speed, stability, and quality of learning. With the changed values, the agent learns faster but may produce different Q-values and average rewards.
