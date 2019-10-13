import numpy as np


def normalise_reward(reward, prev_lives, curr_lives, game_over):
    """
    :param reward: the reward returned by the environment for taking the step.  In space invaders this is the
                   score after the action minus the score before the action.
    :param prev_lives: the lives before the action was taken.
    :param curr_lives: the lives after the action was taken.
    :param game_over: True if the game has ended.
    """
    # Check if we've lost a life, or the game is over, and set the reward accordingly
    if prev_lives != curr_lives:
        reward = -1
    if game_over:
        reward = -1
    if reward > 0:
        reward = 1

    return reward


def get_num_actions():
    """
    Returns the number of actions available in the game;
    (NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE)
    """
    return 6


def get_gym_game_name():
    """
    Get the name of the game as used by OpenAI gym.
    """
    return 'SpaceInvadersNoFrameskip-v4'


def crop_frame(frame):
    """
    Given a game frame, crops the image so only the playable area is in view.
    """
    return frame[9:195,:]


def choose_random_action(env):
    """
    Given the environment, chooses a random action for the agent.
    """
    return env.action_space.sample()


def convert_action_for_env(action):
    """
    No conversion required.
    """
    return action


def get_learning_rate():
    """
    Get the learning rate for gradient descent.
    """
    return 0.00001
