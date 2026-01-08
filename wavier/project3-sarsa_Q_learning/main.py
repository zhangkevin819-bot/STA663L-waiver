import gymnasium as gym
import matplotlib.pyplot as plt
from algorithms import QLearning, Sarsa
from utils import render_single_Q, evaluate_Q
import pandas as pd


# Feel free to run your own debug code in main!
# See https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/taxi.py
def main():
    num_episodes = 5000
    env = gym.make('Taxi-v3')
    # q_learning
    Q1, Q_rewards = QLearning(env, num_episodes)
    evaluate_Q(env, Q1, 200)
    env = gym.make('Taxi-v3', render_mode='human')
    render_single_Q(env, Q1)
    plt.plot(range(num_episodes), Q_rewards)
    plt.show()

    # sarsa
    env = gym.make('Taxi-v3')
    # q_learning
    Q2, Q_rewards = Sarsa(env, num_episodes)
    evaluate_Q(env, Q2, 200)
    env = gym.make('Taxi-v3', render_mode='human')
    render_single_Q(env, Q2)

    plt.plot(range(num_episodes), Q_rewards)
    plt.show()

# 4.3 draw table for Qlearning
    total_reward1 = []
    avg_reward1 = []
    learning_rates = [0.1, 0.2, 0.5, 1.0]
    for lr in learning_rates:
        num_episodes = 5000
        env = gym.make('Taxi-v3')
        gamma = 0.95
        Q11, Q_rewards1 = QLearning(env, num_episodes, gamma, lr,  e=1, decay_rate=0.99)
        total_reward11, avg_reward11= evaluate_Q(env, Q11, 200)
        total_reward1.append(total_reward11)
        avg_reward1.append(avg_reward11)
    df = pd.DataFrame({
        'learning_rates_Qlearning': learning_rates,
        'total_reward_Qlearning': total_reward1,
        'avg_reward_Qlearning': avg_reward1
    })
    print(df)

    total_reward2 = []
    avg_reward2 = []
    learning_rates = [0.1, 0.2, 0.5]
    for lr in learning_rates:
        num_episodes = 5000
        env = gym.make('Taxi-v3')
        gamma = 0.95
        Q22, Q_rewards2 = Sarsa(env, num_episodes, gamma, lr, e=1, decay_rate=0.99)
        total_reward22, avg_reward22 = evaluate_Q(env, Q22, 200)
        total_reward2.append(total_reward22)
        avg_reward2.append(avg_reward22)
    df = pd.DataFrame({
        'learning_rates_Sarsa': learning_rates,
        'total_reward_Sarsa': total_reward2,
        'avg_reward_Sarsa': avg_reward2
    })
    print(df)
if __name__ == '__main__':
    main()
