#this code is a modified version of https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal

import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


from ddpg_agent import Agent

env = gym.make('BipedalWalker-v3')
env.seed(10)
agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=10)

agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

state = env.reset()
agent.reset()   
while True:
    action = agent.act(state)
    env.render()
    next_state, reward, done, _ = env.step(action)
    state = next_state
    if done:
        break
        
env.close()