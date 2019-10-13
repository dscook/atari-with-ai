#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Lambda
from keras.optimizers import Adam
from collections import deque

# To select the game to play ('space-invaders' or 'pong')
from game_selection import game

if game == 'space-invaders':
    from game_space_invaders import choose_random_action, get_learning_rate
elif game == 'pong':
    from game_pong import choose_random_action, get_learning_rate


class Agent:

    def __init__(self, env, num_actions, image_size, num_channels):
        """
        :param env: the space invaders environment.
        :param num_actions: the number of actions the agent can take in the environment.
        :param image_size: the size of a frame in the game.
        :param num_channels: how many frames are stacked together as input to the conv net.
        """
        self.env = env
        self.num_actions = num_actions
        self.state_size = image_size + (num_channels,)
        
        # Note the q_network is not compiled immediately as the client of this class may choose to call
        # the load_weights() method of this class to prime with some already trained weights
        self.q_network = self.create_q_network()
        self.q_target_network = self.create_q_network()
        self.q_target_network.compile(loss='mse', optimizer=Adam(lr=get_learning_rate()))
        self.update_q_target_network_weights()
        
        epsilon_start = 1                   # Explore everything at the beginning
        epilson_end = 0.1                   # Epsilon will decay to this value and remain at it going forwards in training
        self.epsilon_end_frame = 4000000    # The frame at which epsilon should reach its end value
        
        self.epsilon = epsilon_start
        self.epsilon_decay_multiplier = np.exp(np.log(epilson_end/epsilon_start) / self.epsilon_end_frame)
        
        self.discount_factor = 0.99
        
        # To keep track of how many learning steps we have completed
        self.learning_steps = 0
        
        # To maintain the average loss of the CNN during training
        self.average_loss = deque([], maxlen=500)
        

    def create_q_network(self):
        q_network = Sequential()
        q_network.add(Lambda(lambda x: x / 255.0, input_shape=self.state_size))
        q_network.add(Conv2D(32, (8, 8), strides=4, activation='relu'))
        q_network.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
        q_network.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
        q_network.add(Flatten())
        q_network.add(Dense(512, activation='relu'))
        q_network.add(Dense(self.num_actions, activation='linear'))
        return q_network
        
    
    def compile_q_network(self):
        self.q_network.compile(loss='mse', optimizer=Adam(lr=get_learning_rate()))
        
        
    def save_weights(self, file_suffix):
        self.q_network.save_weights('weights/cnn_weights_{}.h5'.format(file_suffix))
    
    
    def load_weights(self, file_suffix):
        self.q_network.load_weights('weights/cnn_weights_{}.h5'.format(file_suffix))


    def select_epsilon_greedy_action(self, state=None):
        if state is None or np.random.uniform(0, 1) < self.epsilon:
            return choose_random_action(self.env)
        else:            
            q_values = self.q_network.predict(state)
            return np.argmax(q_values[0])  # choose greedy action


    def update_q_target_network_weights(self):
        """
        Overwrite the target network weights with the weights from the q network.
        """
        weights = np.asarray(self.q_network.get_weights())
        self.q_target_network.set_weights(weights)


    def learn(self, minibatch):
        """
        Minibatch contains a list of tuples of the form:
            (old_state, action, reward, new_state)
        """          
        # Group together old states and new states to make more efficient predictions
        predict_batch_size = (len(minibatch),) + self.state_size
        old_state_batch = np.empty(shape=predict_batch_size)
        new_state_batch = np.empty(shape=predict_batch_size)
                
        for i in range(len(minibatch)):
            old_state_batch[i] = minibatch[i][0]
            
            # For new states that are None, i.e. game end, just use a random array as we won't use
            # the prediction in any case
            if minibatch[i][3] is None:
                new_state_batch[i] = np.empty(shape=self.state_size)
            else:
                new_state_batch[i] = minibatch[i][3]
        
        # Get Q values for old and new states
        target_q_values = self.q_network.predict(old_state_batch)
        new_state_q_values = self.q_target_network.predict(new_state_batch)
        
        # Update the target Q values
        for i in range(len(minibatch)):
            if minibatch[i][3] is None:
                target_q_values[i, minibatch[i][1]] = minibatch[i][2]
            else:
                target_q_values[i, minibatch[i][1]] = minibatch[i][2] + (
                        self.discount_factor * np.max(new_state_q_values[i]))
        
        history_callback = self.q_network.fit(old_state_batch, target_q_values, epochs=1, batch_size=len(minibatch), verbose=0)
        loss_history = history_callback.history['loss']
        self.average_loss.append(loss_history[0])
                
        # Every X learning steps, update the target network weights
        self.learning_steps += 1
        if self.learning_steps % 2500 == 0:
            self.update_q_target_network_weights()
            
        # Periodically output the training loss
        if len(self.average_loss) == 500 and self.learning_steps % 500 == 0:
            print('Average loss of CNN: {}'.format(np.mean(np.array(self.average_loss))))


    def decay_epsilon(self, frame_number):
        """
        Decay the epsilon greedy parameter so we take less random actions over time.
        """
        # Only decay until a specified frame number
        if frame_number < self.epsilon_end_frame:
            self.epsilon *= self.epsilon_decay_multiplier
