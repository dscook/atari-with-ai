#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import gym
from gym import wrappers
from collections import deque
import matplotlib.pyplot as plt

# My imports
from agent import Agent
from episode_common import process_frames

# To select the game to play ('space-invaders' or 'pong')
from game_selection import game

if game == 'space-invaders':
    from game_space_invaders import get_num_actions, get_gym_game_name, convert_action_for_env
elif game == 'pong':
    from game_pong import get_num_actions, get_gym_game_name, convert_action_for_env


# Game variables
num_actions = get_num_actions()    # Number of actions the agent can take in the environment
image_size = (84, 84)   # Size of the game frames after processing
action_frame_repeat = 4 # The number of frames to repeat an action for
phi_state_size = 4      # The number of processed frames (where an action was selected) to stack to form an input for the CNN


# Runs episodes using CNN weights learnt during training and records the episodes as mp4s
# OR runs several trials for each set of weights to create an averaged learning graph

create_mp4s = False     # Set to false to create averaged learning graph

# When creating average learning graph, number of trials to create average for set of weights
number_of_trials_for_average = 100


# Load each set of weights, weights files are generated periodically as training progresses
frame_matcher = re.compile(r'cnn_weights_(\d+)\.h5')

# To sort the weights in order
weight_frame_number = []

for filename in os.listdir('weights'):
    if filename != '.gitignore':
        match = frame_matcher.match(filename)
        weight_frame_number.append(int(match.group(1)))
        
weight_frame_number = np.array(weight_frame_number)
weight_frame_number = np.sort(weight_frame_number)

# Now run several episodes for each set of weights OR create an mp4 for each set of weights

episode_scores = np.zeros(shape=(len(weight_frame_number), number_of_trials_for_average))
weights_index = 0

for weights in weight_frame_number:
    
    # Record progress
    print('Running episodes for weights at frame {}'.format(weights))
    
    # Create the environment
    env = gym.make(get_gym_game_name())
    if create_mp4s:
        env = wrappers.Monitor(env, './videos/{}'.format(weights), force=True)
    
    # Create the agent
    agent = Agent(env, num_actions, image_size, phi_state_size)
    
    # Set agent randomness to low value
    agent.epsilon = 0.05
    
    # Load the weights into the agent and compile the network
    agent.load_weights(weights)
    agent.compile_q_network()
    
    # Now run either once if recording an MP4 or several trials if creating an average learning graph
    repeats = number_of_trials_for_average
    if create_mp4s:
        repeats = 1
    
    # Run the episodes
    while repeats != 0:
        
        # To store the total score
        total_score = 0
        
        # To record when the episode is complete
        done = False
        
        # Each frame must be combined with the previous frame as lasers blink and we need to do
        # this so we don't miss them
        to_process_frame_buffer = deque([], maxlen=2)
        
        # Maintain the last four processed frames where an action was selected for feeding to the CNN
        state = None
        state_buffer = deque([], maxlen=phi_state_size)
        
        # Restart the game
        observation = env.reset()
        to_process_frame_buffer.appendleft(observation)
        last_processed_frame = process_frames(observation, None, image_size)
                
        # To reduce run time every action will be repeated for four frames.
        # Keep track of how many times we've repeated the action
        action = 0
        action_num_left_to_repeat = action_frame_repeat
        
        # Feed the agent with 4 frames stacked on top of each other
        state = None
        
        # Run the episode until completion
        while not done:
            
            # We only need to choose a new action every 4 frames otherwise
            # repeat the previous action
            if action_num_left_to_repeat > 0:
                action_num_left_to_repeat -= 1
            else:
                # We're about to take an action, record the frame it was taken on
                state_buffer.appendleft(last_processed_frame)
                
                if len(state_buffer) == phi_state_size:
                    # Stack the last 4 processed frames where an action was selected to form the new state
                    state = np.stack(state_buffer, axis=-1)
                                
                # Select the action to take for the next four frames
                if state is None:
                    action = agent.select_epsilon_greedy_action()
                else:
                    action = agent.select_epsilon_greedy_action(state[np.newaxis, ...])
                action_num_left_to_repeat = action_frame_repeat-1
            
            # Take the action
            observation, reward, done, info = env.step(convert_action_for_env(action))
            
            # Maintain the total score for the episode
            total_score += reward
            
            # Process the most recently observed frame
            to_process_frame_buffer.appendleft(observation)
            last_processed_frame = process_frames(to_process_frame_buffer[1], to_process_frame_buffer[0], image_size)
            
            # Store total score for episode
            if done:
                episode_scores[weights_index, number_of_trials_for_average - repeats] = total_score
                    
        repeats -= 1
    
    # Free up resources
    env.close()
    
    # Move along the weights index for recording average learning per set of weights
    weights_index += 1

# Create an average learning graph
if not create_mp4s:
    average_total_scores = episode_scores.mean(axis=1)
    print('Average total score {}, standard deviation {}'.format(average_total_scores, 
          episode_scores.std(axis=1)))
    
    plt.title('Average Total Score vs Frames of Learning')
    plt.ylabel('Average Total Score (over {} episodes)'.format(number_of_trials_for_average))
    plt.xlabel('Frames of Learning (million)')
    plt.plot(weight_frame_number/1e6, average_total_scores)
    plt.savefig('results/average_learning_graph.pdf', bbox_inches='tight')
