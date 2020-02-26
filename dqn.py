# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:05:22 2020

@author: Talha
"""

import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
from util import plot_learning_curve

class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, mode):
        super(LinearDeepQNetwork, self).__init__()
        
        self.mode=mode
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device) #Sending our network to GPU
    
    def forward(self, observation):
        if not self.mode:
            state = T.Tensor(observation).to(self.device)
            layer1 = F.relu(self.fc1(state))
            layer2 = F.relu(self.fc2(layer1))
            actions = self.fc3(layer2)
        else:
            layer1 = F.relu(self.fc1(observation))
            layer2 = F.relu(self.fc2(layer1))
            actions = self.fc3(layer2)
        
        return actions
    
class Agent(object):
    def __init__(self, input_dims, n_actions, lr, batch_size, mode, gamma=0.99, epsilon=1.0, eps_desc=1e-5, eps_min=0.001, max_mem_size=100000):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.eps_desc = eps_desc
        self.eps_min = eps_min
        self.gamma = gamma
        self.action_space = [i for i in range(self.n_actions)]
        self.Q = LinearDeepQNetwork(lr, n_actions, input_dims, mode)
        self.mode = mode
        
        #----------------------For experience replay version----------------------------------------
        if not mode:
            self.batch_size = batch_size
            self.mem_size = max_mem_size
            self.mem_cntr = 0
            self.state_memory = np.zeros((self.mem_size, *input_dims))
            self.new_state_memory = np.zeros((self.mem_size, *input_dims))
            self.action_memory = np.zeros((self.mem_size, self.n_actions), dtype=np.uint8)
            self.reward_memory = np.zeros(self.mem_size)
            self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)
        #-------------------------------------------------------------------------------------------
        
    #--------------------For experience replay version----------------------------------------------   
    def storeTransition(self, state, action, reward, state_,terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        
        #One hot encoding for actions e.g instead of this [0],[1] we will have [0001],[0010] etc.
        #if we get 0 as action(left), 0th index value will be set to 1 or vice versa
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        
        #Fucker breaks!!!
        #self.terminal_memory[index] = terminal
        self.terminal_memory[index] = 1 - terminal
        self.mem_cntr += 1
     #----------------------------------------------------------------------------------------------
     
    def choose_action(self, observation):
       if self.mode: 
            if np.random.random()> self.epsilon:
                state = T.tensor(observation, dtype=T.float).to(self.Q.device)
                actions = self.Q.forward(state)
                action = T.argmax(actions).item()
            else:
                #action = np.random.choice(actions) #checking...
                action = np.random.choice(self.action_space)
            return action
       else:
            actions = self.Q.forward(observation)
            if np.random.random()> self.epsilon:
                action = T.argmax(actions).item()
            else:
                action = np.random.choice(self.action_space)
            return action
            
      
    
    
    def decay_epsilon(self):
        self.epsilon = self.epsilon - self.eps_desc if self.epsilon > self.eps_min else self.eps_min
                            
        
    def learn(self,state, action, reward, state_):
        if self.mode:
         #----------------------------------------------------Naive version---------------------------------------------------------------
            self.Q.optimizer.zero_grad() #Zero our gradients      
             #Datatype conversion for cuda tensors (From numpy arrays)
            states = T.tensor(state, dtype=T.float).to(self.Q.device)
            actions = T.tensor(action).to(self.Q.device)
            #rewards = T.tensor(reward).to(self.Q.device)
            states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)
            
            q_pred = self.Q.forward(states)[actions]
            q_next = self.Q.forward(states_).max()
            q_target = reward + self.gamma*q_next
             
            loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
            loss.backward()
            self.Q.optimizer.step()
            self.decay_epsilon()
         #---------------------------------------------------------------------------------------------------------------------------------
        
        
        #-----------------------------------------------Experience replay version----------------------------------------------------------
        #For the first 64 iterations the agent will not learn from its memory but will explore the action space and create memories once we
        #have exceeded that it will start looking up its memories and will learn from them
        else:
            if self.mem_cntr > self.batch_size:
                self.Q.optimizer.zero_grad() #Zero our gradients
    
                max_mem = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size
    
                #Getting our agents memories randomly and feeding it to our feedforward network for getting the expected reward estimates
                batch = np.random.choice(max_mem, self.batch_size)
                
                #getting the state from the memory
                state_batch = self.state_memory[batch]
                
                #getting the indexes from the memories for the current action
                action_batch = self.action_memory[batch]
                action_values = np.array(self.action_space, dtype=np.int32)
                action_indices = np.dot(action_batch, action_values)
                
                #getting the reward,S',done flag
                reward_batch = self.reward_memory[batch]
                new_state_batch = self.new_state_memory[batch]
                terminal_batch = self.terminal_memory[batch]
                
                #As the network returns a tensor so in order to perform any arthematic operation we need to convert our data to tensor
                reward_batch = T.Tensor(reward_batch).to(self.Q.device)
                terminal_batch = T.Tensor(terminal_batch).to(self.Q.device)
                
                q_pred = self.Q.forward(state_batch).to(self.Q.device)
                #We are using the same network for getting the q_target which we will later update with our bellman equation
                #q_target = self.Q.forward(state_batch).to(self.Q.device)
                #instead of running the network again for getting the q_target, we will just use the clone() 
                q_target = q_pred.clone()
                q_next = self.Q.forward(new_state_batch).to(self.Q.device)
                
                batch_index = np.arange(self.batch_size, dtype=np.int32)
                q_target[batch_index, action_indices] = reward_batch + self.gamma * T.max(q_next, dim=1)[0]*terminal_batch
                
                
                loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
                loss.backward()
                self.Q.optimizer.step()
                self.decay_epsilon()
        #------------------------------------------------------------------------------------------------------------------------------------
        
        
if __name__=='__main__':
    env = gym.make('CartPole-v1')
    n_episodes = 5000
    scores = []
    eps_history = []
    score = 0
    agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.n, lr=0.001, batch_size=64, mode=True) #Mode=False:Experience Replay else Naive
    
    for i in range(n_episodes):
        score = 0
        done = False
        obs = env.reset()
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            if not agent.mode:
                agent.storeTransition(obs, action, reward, obs_, done)
            agent.learn(obs, action, reward, obs_)
            obs = obs_
            
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        if i % 100==0:
            avg_score = np.mean(scores[-100:])
            print('episode',i,'score %.1f avg score %.1f epsilon %.2f' %(score, avg_score, agent.epsilon))
            
    filename='cartpole_dqn_eps_min_001.png'
    x = [i+1 for i in range(n_episodes)]
    plot_learning_curve(x, scores, eps_history, filename)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    