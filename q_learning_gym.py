import numpy as np
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0',  is_slippery=False)

# action enum
class Actions:
    Left = 0
    Down = 1
    Right = 2
    Up = 3

# init vars
n_episodes = 20000
wins = 0
win_history = []
reward_movement = 0
reward_history = []
alpha = 0.1
gamma = 0.96
epsilon = 0.9
min_epsilon = 0.01
decay_rate = 0.9
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# q learning loop
for i in tqdm(range(n_episodes)):
    state = env.reset()
    action = None
    done = False

    # chance of taking a random action decreases as episodes progress
    if i % 100 == 99:
        epsilon *= decay_rate
        epsilon = np.max([epsilon, min_epsilon])

    z = 0
    while z < 100:
        if action == None:
            action = env.action_space.sample()
        else:
            # sometimes take a random action
            #action = env.action_space.sample()
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

        new_state, reward, done, _ = env.step(action)

        # q values updated backwards from rewards
        # we set the new q value for the chosen action based on the max state_action value of the new state
        old_q_val = q_table[state, action]
        new_q_val = old_q_val + alpha * (reward + gamma * np.max(q_table[new_state, :]) - old_q_val)
        q_table[state, action] = new_q_val

        state = new_state

        z += 1

        if done: 
            if reward > 0:
                wins += 1
                reward_movement += 1
            else:
                reward_movement -= 1
            
            reward_history.append(reward_movement)
            break

    win_history.append(wins)
            
plt.plot(reward_history)

# play a game
import time
env.reset()
done = False
state = 0
while not done:
    state, reward, done, _ = env.step(np.argmax(q_table[state, :]))
    env.render()
    time.sleep(0.5)


