' Reinforcement Learning: Q learning'

# import libraries
import numpy as np
import random

# define the enviroment (4x4 grid)
num_states = 16 # 4x4 grid
num_actions = 4 # Up(0), Right(1), Down(2), Left(3)
q_table = np.zeros((num_states, num_actions))

# define the parameters
alpha = 0.1 # learning rate
gamma = 0.9 # discount factor
epsilon = 0.2 # exploration rate
num_episodes = 1000

# define a simple reward structure
rewards = np.zeros(num_states)
rewards[15] = 1 # Goal state with a reward

# function to determine the next state based on the action
def get_next_state(state, action):
    if action == 0 and state >= 4: # up
        return state - 4
    elif action == 1 and (state + 1) % 4 != 0: # right
        return state + 1
    elif action == 2 and state < 12: # down
        return state + 4
    elif action == 3 and state % 4 != 0: # left
        return state - 1
    else:
        return state
    
# Q learning algorithm
for episode in range(num_episodes):
    state = random.randint(0, num_states - 1) # start from random state
    while state != 15:  # loop until reaching the goal state
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, num_actions - 1)# random action
        else:
            action = np.argmax(q_table[state])
        
        next_state = get_next_state(state, action)
        reward = rewards[next_state]
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # Q learning update rule
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[state, action] = new_value

        state = next_state # move to the next state

# display the learned Q-table
print("Learned Q-Table:")
print(q_table)