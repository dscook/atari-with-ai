#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

# To select the game to play ('space-invaders' or 'pong')
from game_selection import game

if game == 'space-invaders':
    from game_space_invaders import crop_frame
elif game == 'pong':
    from game_pong import crop_frame

i=0

def process_frames(first_frame, second_frame, image_size):
    """
    Given two frames in sequence:
        1. Converts them both to greyscale.
        2. Takes the maxiumum intensity pixel across both frames to account for the fact lasers in the game
           flash and it we don't do this we won't always see them.
        3. Crops the image to remove uninformative parts of the screen (to increase deep learning speed).
        4. Reduce the size of the image (to increase deep learning speed).
        
    If given only one frame will convert it to greyscale, crop and resize.
    
    :returns: processed frame of shape (84, 84).
    """
    #plt.imsave('1.png', first_frame)
    #plt.imsave('2.png', second_frame)
    
    # Convert frame to greyscale
    first_frame_grey = 0.2126 * first_frame[:,:,0] + 0.7152 * first_frame[:,:,1] + 0.0722 * first_frame[:,:,2]
    first_frame_grey = first_frame_grey.astype(np.uint8)
    
    # Crop to remove uninformative areas of the screen    
    cropped_first = crop_frame(first_frame_grey)
    
    cropped_second = None
    if second_frame is not None:
        second_frame_grey = 0.2126 * second_frame[:,:,0] + 0.7152 * second_frame[:,:,1] + 0.0722 * second_frame[:,:,2]
        second_frame_grey = second_frame_grey.astype(np.uint8)
        cropped_second = crop_frame(second_frame_grey)

    # Take the max intensity pixel for each cell across images
    combined = None
    if second_frame is not None:
        combined = np.maximum(cropped_first, cropped_second)
    else:
        combined = cropped_first
    
    # Reduce the size of the image and hence increase deep learning speed
    #resized_combined = combined
    resized_combined = np.array(Image.fromarray(combined).resize(image_size, resample=Image.BICUBIC))
    
    #global i
    #plt.imsave('resized{}.png'.format(i), resized_combined, cmap='gray', vmin=0, vmax=255)
    #i += 1
    
    return resized_combined