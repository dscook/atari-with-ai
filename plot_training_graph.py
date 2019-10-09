#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt

training_to_plot = '1555420000'

filename = 'results/training_results_{}.csv'.format(training_to_plot)

with open(filename, 'r') as csv_file:    
    
    results_dict = csv.DictReader(csv_file)
    
    # Collect the total scores
    episode_number = []
    total_scores = []
    for row in results_dict:
        episode = int(row['episode'])
        episode_number.append(episode)
        total_score = int(float(row['total score']))
        total_scores.append(total_score)
        
    # Average a SMALL sliding window across episodes to smooth the learning curve
    total_scores = np.array(total_scores)
    window_size = 20
    total_scores[:] = [
            (np.sum(total_scores[i-window_size+1:i+1]) / window_size) if i>(window_size-1)
            else (np.sum(total_scores[:i+1]) / (i+1)) for i in range(len(total_scores))]
    
    plt.title('Total Score vs Episodes Played')
    plt.ylabel('Total Score ({} Episode Average)'.format(window_size))
    plt.xlabel('Episodes Played')
    plt.plot(np.array(episode_number)+1, total_scores)
    plt.savefig('results/training_graph.pdf', bbox_inches='tight')
    plt.show()
        
