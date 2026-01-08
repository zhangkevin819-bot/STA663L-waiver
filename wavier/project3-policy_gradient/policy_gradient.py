import gymnasium as gym
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.distributions import Categorical
import torch.nn as nn
import numpy as np

# see https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
# understand environment, state, action and other definitions first before your dive in.

# uncomment to switch between environments
ENV_NAME = 'CartPole-v0'
#ENV_NAME = 'CartPole-v1'

# Hyper Parameters
# Following params work well if your implement Policy Gradient correctly.
# You can also change these params.
EPISODE = 3000  # total training episodes
STEP = 500 if ENV_NAME == 'CartPole-v1' else 200  # step limitation in an episode
EVAL_EVERY = 10  # evaluation interval
TEST_NUM = 5  # number of tests every evaluation
GAMMA = 0.95  # discount factor
LEARNING_RATE = 3e-3  # learning rate for mlp


# A simple mlp implemented by PyTorch #
# it receives (N, D_in) shaped torch arrays, where N: the batch size, D_in: input state dimension
# and outputs the possibility distribution for each action and each sample, shaped (N, D_out)
# e.g. 
# state = torch.randn(10, 4)
# outputs = mlp(state)  #  output shape is (10, 2) in CartPole-v0 Game
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

class REINFORCE:
    def __init__(self, env):
        # init parameters
        self.time_step = 0
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.states, self.actions, self.action_probs, self.rewards = [], [], [], []
        self.last_state = None
        self.net = MLP(input_dim=self.state_dim, output_dim=self.action_dim)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

    def predict(self, observation, deterministic=False):
        observation = torch.FloatTensor(observation).unsqueeze(0)
        action_score = self.net(observation)
        probs = F.softmax(action_score, dim=1)
        m = Categorical(probs)
        if deterministic:
            action = torch.argmax(probs, dim=1)
        else:
            action = m.sample()
        return action, probs

    def store_transition(self, s, a, p, r):
        self.states.append(s)
        self.actions.append(a)
        self.action_probs.append(p)
        self.rewards.append(r)

    def learn(self):
        # Please make sure all variables used to calculate loss are of type torch.Tensor, or autograd may not work properly.
        # You need to calculate the loss of each step of the episode and store them in '''loss'''.
        # The variables you should use are: self.rewards, self.action_probs, self.actions.
        # self.rewards=[R_1, R_2, ...,R_T], self.actions=[A_0, A_1, ...,A_(T-1)]
        # self.action_probs corresponds to the probability of different actions of each timestep, see predict() for details

        loss = []
        # -------------------------------
        # Your code goes here
        rewards_new = []
        a = 0
        for b in reversed(self.rewards):
            a = b + GAMMA * a
            rewards_new.insert(0, a)
        rewards_new = torch.tensor(rewards_new)
        rewards_new = (rewards_new - rewards_new.mean()) / (rewards_new.std())
        for prob, reward in zip(self.action_probs, rewards_new):
            loss.append(-prob[0, self.actions.pop(0)] * reward)
        # TODO Calculate the loss of each step of the episode and store them in '''loss'''
        # -------------------------------

        # code for autograd and back propagation
        self.optim.zero_grad()
        loss = torch.cat(loss).sum()
        loss.backward()
        self.optim.step()

        self.states, self.actions, self.action_probs, self.rewards = [], [], [], []
        return loss.item()

#M=101
M=88
#M=188
def main():
    # initialize OpenAI Gym env and PG agent
    env = gym.make(ENV_NAME)
    env_render = gym.make(ENV_NAME, render_mode='human')
    # uncomment to switch between methods

    agent = REINFORCE(env)
    E_reward =[]
    E_loss = []

    env.action_space.seed(M)
    env.observation_space.seed(M)

    env_render.action_space.seed(M)
    env_render.observation_space.seed(M)
    for episode in range(EPISODE):
        # initialize task
        state, info = env.reset()
        agent.last_state = state
        rewardf = 0

        # Train
        for step in range(STEP):
            action, probs = agent.predict(state)
            next_state, reward, done, truncated, _ = env.step(action.item())
            agent.store_transition(state, action, probs, reward)
            state = next_state
            rewardf += reward
            if done:
                loss = agent.learn()
                break
        E_reward.append(rewardf)
        E_loss.append(loss)


        # Test
        if (episode + 1) % EVAL_EVERY == 0:
            total_reward = 0
            env_test = env
            env_test = env_render
            for i in range(TEST_NUM):
                state, info = env_test.reset()
                for j in range(STEP):
                    # You may uncomment the line below to enable rendering for visualization.
                    action, _ = agent.predict(state, deterministic=True)
                    state, reward, done, truncated, _ = env_test.step(action.item())
                    total_reward += reward
                    if done:
                        break
            avg_reward = total_reward / TEST_NUM

            # Your avg_reward should reach 200(cartpole-v0)/500(cartpole-v1) after a number of episodes.
            print('episode: ', episode, 'Evaluation Average Reward:', avg_reward)

    plt.plot(range(EPISODE), E_reward)
    plt.show()

    plt.plot(range(EPISODE), E_loss)
    plt.show()


if __name__ == '__main__':
    main()
