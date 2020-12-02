import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch

from torch import nn
from torch import optim
from torch import tensor

from bm_multi_env import Game

class PolicyEstimator():
    def __init__(self):

        self.num_observations = 5 * 7
        self.num_actions = 6

        # self.network = nn.Sequential(
        #     nn.Linear(self.num_observations, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, self.num_actions),
        #     nn.Softmax(dim=-1)
        # )

        self.network = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1120, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
            nn.Softmax(dim=-1)
        )

    def predict(self, observation):
        observation = torch.tensor(observation).float()
        return self.network(observation)



def preprocess(image):
    """ Pre-process 210x160x3 uint8 frame into 6400 (80x80) 1D float vector. """
    image = torch.tensor(image)
    ii = torch.tensor(np.expand_dims(np.array([image.numpy() == bt for bt in range(10)]), 0)).float()
    return ii

# debug 
#num_episodes=1500; batch_size=10; discount_factor=0.99; render=False; early_exit_reward_amount=None; env = Game(5,7); estimator = PolicyEstimator()

def vanilla_policy_gradient(env, estimator, num_episodes=1500, batch_size=10, discount_factor=0.99, render=False, early_exit_reward_amount=None):
    total_rewards, batch_rewards, batch_observations, batch_actions = [], [], [], []
    batch_counter = 1

    optimizer = optim.Adam(estimator.network.parameters(), lr=0.001)
    action_space = np.arange(6) # [0, 1] for cartpole (either left or right)

    for current_episode in range(num_episodes):
        observation, players = env.reset()
        observation = preprocess(observation)
        rewards, actions, observations = [], [], []
        turn = 0
        while True:
            if render:
                env.render()

            # use policy to make predictions and run an action
            action_probs = estimator.predict(observation).squeeze().detach().numpy()
            action = np.random.choice(action_space, p=action_probs) # randomly select an action weighted by its probability
            # if np.random.uniform() < np.exp(-current_episode*0.001):
            #     action = np.random.choice(action_space)
            # else:
            #     action = np.random.choice(action_space, p=action_probs) # randomly select an action weighted by its probability

            # push all episodic data, move to next observation
            observations.append(observation)
            score_before = players[0].score
            observation, done, players, bombs = env.step([action, 0])
            observation = preprocess(observation)
            score_after = players[0].score

            reward = (score_after - score_before)# - np.sum(env.board == 3)

            if reward > 0:
                print("ATTAINED REWARD!!!!!!!!")

            # # don't blow yourself up on the first move
            # if turn == 0 and action == 5:
            #     reward -= 100

            # # don't do nothing
            # if action == 0:
            #     reward -= 1

            rewards.append(reward)
            actions.append(action)

            turn += 1

            if done:
                # apply discount to rewards
                r = np.full(len(rewards), discount_factor) ** np.arange(len(rewards)) * np.array(rewards)
                r = r[::-1].cumsum()[::-1]
                discounted_rewards = r - r.mean()

                # collect the per-batch rewards, observations, actions
                batch_rewards.extend(discounted_rewards)
                batch_observations.extend(observations)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                if batch_counter >= batch_size:
                    # reset gradient
                    optimizer.zero_grad()

                    # tensorify things
                    batch_rewards = torch.tensor(batch_rewards).float()
                    batch_observations = torch.stack(batch_observations, 1).squeeze() #torch.tensor(batch_observations).float()
                    batch_actions = torch.tensor(batch_actions).long()

                    # calculate loss
                    logprob = torch.log(estimator.predict(batch_observations))
                    batch_actions = batch_actions.reshape(len(batch_actions), 1)
                    selected_logprobs = batch_rewards * torch.gather(logprob, 1, batch_actions).squeeze()
                    loss = -selected_logprobs.mean()

                    # backprop/optimize
                    loss.backward()
                    for param in estimator.network.parameters():
                        param.grad.data.clamp_(-1, 1)
                    optimizer.step()

                    # reset the batch
                    batch_rewards, batch_observations, batch_actions = [], [], []
                    batch_counter = 1

                # get running average of last 100 rewards, print every 100 episodes
                average_reward = np.mean(total_rewards[-100:])
                if current_episode % 100 == 0:
                    print(f"average of last 100 rewards as of episode {current_episode}: {average_reward:.2f}")

                # quit early if average_reward is high enough
                if early_exit_reward_amount and average_reward > early_exit_reward_amount:
                    return total_rewards

                break

    return total_rewards

#if __name__ == '__main__':
# create environment
#env_name = 'CartPole-v0'
#env = gym.make(env_name)

env = Game(5,7)
estimator = PolicyEstimator()
# actually run the algorithm
rewards = vanilla_policy_gradient(env, estimator, num_episodes=70000)

# moving average
moving_average_num = 100
def moving_average(x, n=moving_average_num):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)

# plotting
plt.scatter(np.arange(len(rewards)), rewards, label='individual episodes')
plt.plot(moving_average(rewards), label=f'moving average of last {moving_average_num} episodes')
plt.title(f'Vanilla Policy Gradient on bm')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()




env = Game(5,7)
observation, p = env.reset()
observation = observation.flatten()
preds = estimator.predict(observation)
observation, d,p,b = env.step()