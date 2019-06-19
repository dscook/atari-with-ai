#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# Experience replay buffer size.  Need to store current and next state in buffer for each entry so ...
# Size ~ 2 * image size * frames repeated * 1 byte for each greyscale value * replay buffer size
replay_buffer_size = 100000     # (2 * 84 * 84 * 4 * 1 * 100000 ~ 6GB)
#replay_buffer_size = 200000   # AWS (2 * 84 * 84 * 4 * 1 * 200000 ~ 12GB)


class ReplayBuffer:


    def __init__(self, mini_batch_size):
        self.replay_buffer = []
        self.replay_buffer_count = 0
        self.mini_batch_size = mini_batch_size
        self.oldest_index = 0


    def add(self, old_state, action, reward, new_state):
        """
        If replay buffer not yet full just add to the end of the buffer otherwise
        replace the oldest element with this element.
        new_state will be None if the game terminates after taking action in old_state.
        """        
        if self.replay_buffer_count < replay_buffer_size:
            self.replay_buffer.append((old_state, action, reward, new_state))
            self.replay_buffer_count += 1
        else:
            self.replay_buffer[self.oldest_index] = (old_state, action, reward, new_state)
            self.oldest_index += 1
            if self.oldest_index == replay_buffer_size:
                self.oldest_index = 0
    
    
    def get_random_minibatch(self):
        """
        Returns a minibatch of states from the replay buffer, returns None if the replay buffer
        does not have enough data to create a full minibatch.
        """
        if self.replay_buffer_count < self.mini_batch_size:
            return None
        else:
            mini_batch_indexes = np.random.randint(0, self.replay_buffer_count, self.mini_batch_size)
            mini_batch = []
            for index in mini_batch_indexes:
                mini_batch.append(self.replay_buffer[index])
            return mini_batch


    def is_full(self):
        """
        Returns true if the replay buffer is full.
        """
        return self.replay_buffer_count == replay_buffer_size