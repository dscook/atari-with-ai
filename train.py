#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import numpy as np
from collections import deque
from time import time

# My imports
from agent import Agent
from episode_common import process_frames
from replay_buffer import ReplayBuffer

# To select the game to play ('space-invaders' or 'pong')
from game_selection import game

if game == 'space-invaders':
    from game_space_invaders import normalise_reward, get_num_actions, get_gym_game_name, convert_action_for_env
elif game == 'pong':
    from game_pong import normalise_reward, get_num_actions, get_gym_game_name, convert_action_for_env


# Training variables
frames_to_train = 40000000
save_weights_interval = frames_to_train/10        # How often to save neural network weights to reproduce agent

# Game variables
num_actions = get_num_actions()    # Number of actions the agent can take in the environment
image_size = (84, 84)   # Size of the game frames after processing
action_frame_repeat = 4 # The number of frames to repeat an action for
phi_state_size = 4      # The number of processed frames (where an action was selected) to stack to form an input for the CNN

# Learning rate variables
mini_batch_size = 32
min_experience_frame_limit = 200000  # Do not begin learning until this many frames have been experienced
cnn_learning_interval = 4            # Only update CNN weights after the agent has selected actions four times


# Version of space invaders environment that proceeds one frame at a time.
# Also the game is deterministic, the agent will take every action we ask
# it to take in the game and not occassionaly do something random (unless we code it).
env = gym.make(get_gym_game_name())

# A file to maintain the results of training, flush after every line
results_file = open('results/training_results_{}.csv'.format(int(time())), 'a', buffering=1)
results_file.write('episode,frame,total reward,total score\n')

# To maintain how many frames and episodes we have completed so far
frame_number = 0
episode_number = 0

# Create the agent to learn
agent = Agent(env, num_actions, image_size, phi_state_size)
agent.compile_q_network()

# Experience replay buffer
replay_buffer = ReplayBuffer(mini_batch_size)

# Train for at least the required number of frames
while frame_number < frames_to_train:
            
    # Store the total reward for the episode as well as the actual game score
    total_reward = 0
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
    frame_number += 1
    
    # Store the number of lives we have
    num_lives = 3
    
    # To reduce run time every action will be repeated for four frames.
    # Keep track of how many times we've repeated the action
    # Take no action for the first set of frames
    action = 0
    action_num_left_to_repeat = action_frame_repeat
    reward_from_action = 0
    
    # To keep track of when we are due to update the CNN weights
    action_selects_until_weight_update = cnn_learning_interval
        
    # Run the episode until completion
    while not done:
        
        if (frame_number-1) % save_weights_interval == 0:
            agent.save_weights(frame_number-1)
                        
        # We only need to choose a new action every 4 frames otherwise
        # repeat the previous action
        if action_num_left_to_repeat > 0:
            action_num_left_to_repeat -= 1
        else:
            # We're about to take an action, record the frame it was taken on
            state_buffer.appendleft(last_processed_frame)
            
            if len(state_buffer) == phi_state_size:
                # Stack the last 4 processed frames where an action was selected to form the new state
                new_state = np.stack(state_buffer, axis=-1)
                
                if state is not None:
                    # Record previous state, new state, action taken and reward observed in the replay buffer
                    replay_buffer.add(state, action, reward_from_action, new_state)
                
                # Get the agent to learn if we've experienced enough random actions
                if frame_number > min_experience_frame_limit:
                    
                    # Only update CNN weights if we are due to
                    if action_selects_until_weight_update == 0:
                        minibatch = replay_buffer.get_random_minibatch()
                        agent.learn(minibatch)
                        action_selects_until_weight_update = cnn_learning_interval
                    else:
                        # Agent will be selecting an action, coundown to updating the CNN weights
                        action_selects_until_weight_update -= 1
                
                # New state becomes old state
                state = new_state
                            
            # Select the action to take for the next four frames, act randomly if we've not primed the replay buffer
            if state is None or frame_number <= min_experience_frame_limit:
                action = agent.select_epsilon_greedy_action()
            else:
                action = agent.select_epsilon_greedy_action(state[np.newaxis, ...])
            action_num_left_to_repeat = action_frame_repeat-1
            reward_from_action = 0
                
        # Take the action
        observation, reward, done, info = env.step(convert_action_for_env(action))
        
        # Maintain the total score for the episode before we normalise the reward
        total_score += reward
        
        # Normalise the reward and store the current number of lives
        reward = normalise_reward(reward, num_lives, info['ale.lives'])
        reward_from_action += reward
        total_reward += reward
        num_lives = info['ale.lives']
        
        # Process the most recently observed frame
        to_process_frame_buffer.appendleft(observation)
        last_processed_frame = process_frames(to_process_frame_buffer[1], to_process_frame_buffer[0], image_size)
        frame_number += 1
        agent.decay_epsilon(frame_number)
                        
        # Print and save the total score if the episode has ended
        if done:
            # Ensure we capture the game end new state in the replay buffer
            # to capture the negative reward for loss of life
            if state is not None:
                replay_buffer.add(state, action, reward_from_action, None)
            
            print('Episode {}, Frame {}, Total Score {}, Epsilon {}'.format(episode_number, frame_number, total_score, agent.epsilon))
            results_file.write('{},{},{},{}\n'.format(episode_number, frame_number, total_reward, total_score))
            episode_number += 1

# Free up resources
env.close()
results_file.close()