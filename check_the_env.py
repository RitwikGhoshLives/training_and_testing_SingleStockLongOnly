# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 20:28:57 2023

@author: Ritwik-Ghosh
"""


import gym
import time

train_data_dir = 'stock_data' 

stock_list = ['APOLLOTYRE', 'TATAMOTORS', 'MARUTI', 'HEROMOTOCO', 'CANBK',
              'ICICIBANK', 'SBIN', 'AXISBANK', 'JSWSTEEL', 'TATASTEEL',
              'HINDALCO', 'AUROPHARMA', 'GLENMARK', 'DIVISLAB', 'DRREDDY',
              'WIPRO', 'INFY', 'TCS', 'HCLTECH', 'JUBLFOOD', 'RELIANCE',
              'BHARTIARTL', 'BAJFINANCE', 'DLF']

env = gym.make('SingleStockLongOnlyIndiaEnv',
               list_of_stock = stock_list,
               data_dir = train_data_dir,
               episode_length_in_steps = 10,
               time_stamp_in_min = 5,
               look_back_window = 360,
               initial_fund_in_money = 100000,
               fee_percentage = 0.05,
               max_fee = 20,
               inflation_percentage_per_year = 10,
               max_acceptable_drop_down_percentage_episode_stopping = 5)    

print('\nObservation Sample\n', env.observation_space.sample())
print('\nAction Sample\n', env.action_space.sample())

episodes = 1
final_PandL = 0
for episode in range(episodes):
    start = time.time()
    done = False
    obs = env.reset()
    score = 0
    while not done:
        random_action = env.action_space.sample()
        print('Action taken by the agent:',random_action)
        obs, reward, done, info = env.step(random_action)
        print('Reward at the step:', reward)
    end = time.time()
    print(f"Time taken for the episode: {(end-start)*10**3:.03f}ms")