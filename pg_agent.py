"""
Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym.
Modified from https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
"""

import argparse
import datetime
import os
import random
import time
from bm_multi_env import Game

import gym
import torch
import torch.nn as nn
from IPython.display import clear_output
import torch.nn.functional as F
import numpy as np

def preprocess(image, flatten=True):
    """ Pre-process 210x160x3 uint8 frame into 6400 (80x80) 1D float vector. """
    image = torch.tensor(image)
    if flatten:
        ii = image.flatten().float()
    else:
        ii = torch.tensor(np.expand_dims(np.array([image.numpy() == bt for bt in range(10)]), 0)).float()
    return ii


def calc_discounted_future_rewards(rewards, discount_factor):
    r"""
    Calculate the discounted future reward at each timestep.
    discounted_future_reward[t] = \sum_{k=1} discount_factor^k * reward[t+k]
    """

    discounted_future_rewards = torch.empty(len(rewards))

    # Compute discounted_future_reward for each timestep by iterating backwards
    # from end of episode to beginning
    discounted_future_reward = 0
    for t in range(len(rewards) - 1, -1, -1):
        # If rewards[t] != 0, we are at game boundary (win or loss) so we
        # reset discounted_future_reward to 0 (this is pong specific!)
        # if rewards[t] != 0:
        #     discounted_future_reward = 0

        discounted_future_reward = rewards[t] + discount_factor * discounted_future_reward
        discounted_future_rewards[t] = discounted_future_reward

    return discounted_future_rewards


class PolicyNetwork0(nn.Module):
    """ Simple two-layer MLP for policy network. """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 6)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)

        x = self.fc2(x)
        prob_up = torch.sigmoid(x)

        return prob_up
    
    
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        linear_input_size = 5 * 7 * 32
        self.lin1 = nn.Linear(linear_input_size, 6)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = torch.flatten(x)
        x = self.lin1(x)
        return torch.sigmoid(x)


def run_episode(model, env, discount_factor, render=False):
    LEFT=1
    RIGHT=2
    UP = 3
    DOWN = 4
    BOMB = 5
    NOTHING = 0

    observation, players = env.reset()

    action_chosen_log_probs = []
    rewards = []

    done = False
    timestep = 0

    while not done:
        # Preprocess the observation, set input to network to be difference
        # image between frames
        x = preprocess(observation, False)

        # Run the policy network and sample action from the returned probability
        action_probs = model(x)
        action_pmax = torch.max(action_probs)
        action_amax = torch.argmax(action_probs)
        action_rand = torch.randperm(6)[0]
        action_rand_prob = action_probs[action_rand]
        action = action_amax if random.random() < action_pmax else  action_rand# roll the dice!

        # Calculate the probability of sampling the action that was chosen
        action_chosen_prob = action_pmax if action == action_amax else action_rand_prob
        action_chosen_log_probs.append(torch.log(action_chosen_prob))

        # Step the environment, get new measurements, and updated discounted_reward
        pre_score = players[0].score
        observation, done, players, info = env.step([action.item(), 0])
        post_score = players[0].score
        reward = post_score - pre_score
        
        # apply blocks left penelty
        blocks_left = np.sum(env.board == 3)
        reward = reward - (blocks_left)
        
        rewards.append(torch.Tensor([reward]))
        timestep += 1

    # Concat lists of log probs and rewards into 1-D tensors
    action_chosen_log_probs = torch.stack(action_chosen_log_probs).view(-1, 1)
    rewards = torch.cat(rewards).view(-1, 1)

    # Calculate the discounted future reward at each timestep
    discounted_future_rewards = calc_discounted_future_rewards(rewards, discount_factor)

    # Standardize the rewards to have mean 0, std. deviation 1 (helps control the gradient estimator variance).
    # It encourages roughly half of the actions to be rewarded and half to be discouraged, which
    # is helpful especially in beginning when positive reward signals are rare.
    discounted_future_rewards = (discounted_future_rewards - discounted_future_rewards.mean()) / discounted_future_rewards.std()

    # PG magic happens right here, multiplying action_chosen_log_probs by future reward.
    # Negate since the optimizer does gradient descent (instead of gradient ascent)
    loss = -(discounted_future_rewards * action_chosen_log_probs).sum()

    return loss, rewards.sum()


def train(render=False):
    # Hyperparameters
    input_size = 5 * 7 # input dimensionality: 80x80 grid
    hidden_size = 200 # number of hidden layer neurons
    learning_rate = 7e-5
    discount_factor = 0.99 # discount factor for reward

    batch_size = 16
    save_every_batches = 5

    # Create policy network
    model = PolicyNetwork(input_size, hidden_size)

    # Load model weights and metadata from checkpoint if exists
    if os.path.exists('checkpoint.pth'):
        print('Loading from checkpoint...')
        save_dict = torch.load('checkpoint.pth')

        model.load_state_dict(save_dict['model_weights'])
        start_time = save_dict['start_time']
        last_batch = save_dict['last_batch']
    else:
        start_time = datetime.datetime.now().strftime("%H.%M.%S-%m.%d.%Y")
        last_batch = -1

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create pong environment (PongDeterministic versions run faster)
    env = Game(5,7)

    # Pick up at the batch number we left off at to make tensorboard plots nicer
    batch = last_batch + 1
    while True:

        mean_batch_loss = 0
        mean_batch_reward = 0
        for batch_episode in range(batch_size):

            # Run one episode
            loss, episode_reward = run_episode(model, env, discount_factor, render)
            mean_batch_loss += loss / batch_size
            mean_batch_reward += episode_reward / batch_size

            # Boring book-keeping
            # print(f'Episode reward total was {episode_reward}')

        # Backprop after `batch_size` episodes
        optimizer.zero_grad()
        mean_batch_loss.backward()
        optimizer.step()

        # Batch metrics and tensorboard logging
        #if batch % 1 == 0 and batch > 0:
        clear_output()
        print(f'Batch: {batch}, mean loss: {mean_batch_loss:.2f}, '
              f'mean reward: {mean_batch_reward:.2f}')


        # if batch % save_every_batches == 0:
        #     print('Saving checkpoint...')
        #     save_dict = {
        #         'model_weights': model.state_dict(),
        #         'start_time': start_time,
        #         'last_batch': batch
        #     }
        #     torch.save(save_dict, 'checkpoint.pth')

        batch += 1


def main():
    # By default, doesn't render game screen, but can invoke with `--render` flag on CLI
    train(render=False)


if __name__ == '__main__':
    main()
    
