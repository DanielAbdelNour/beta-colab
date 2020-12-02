import importlib
import gym_bm_multi_env
importlib.reload(gym_bm_multi_env)
from gym_bm_multi_env import Game
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecFrameStack
import gym
import numpy as np
import time
import matplotlib.pyplot as plt


#gym_env = gym.make('FrozenLake-v0', map_name=None)


class A:
	nothing = 0
	left = 1
	right = 2
	up = 3
	down = 4
	bomb = 5

env = Game(5,7)

env.observation_space = gym.spaces.Box(low=0, high=9, shape=(5, 7), dtype=np.uint8)
env.action_space = gym.spaces.Discrete(6)

check_env(env)



# env.reset()
# env.step([A.right, 0])
# env.render()

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("a2c_dd")

model = A2C.load("a2c_dd", verbose=1)

obs = env.reset()

action, _ = model.predict(obs)
print("Action: ", action)
obs, reward, done, info = env.step(action)
print('obs=', obs, 'reward=', reward, 'done=', done)
env.render()


# Test the trained agent
obs = env.reset()
n_steps = 200
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=False)
    #print("Step {}".format(step + 1))
    #print("Action: ", action)
    obs, reward, done, info = env.step(action)
    print('reward=', reward, 'done=', done)
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break

        
        
        
        
env = Game(5, 7)
env.reset()
action, _ = model.predict(obs)
print(action)
obs, reward, done, info = env.step(action)
env.render()