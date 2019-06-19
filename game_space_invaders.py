import numpy as np

# For normalising of reward
# Note loss of life set to -305 and mothership assumed to be 200, the latter isn't always true as 
# the number of points awarded often varies. (5,10,15,20,25,30) for aliens hit, 0 for nothing happened.
# Mean of rewards possible is therefore 0.
rewards_possible = [-305, 0, 5, 10, 15, 20, 25, 30, 200]
reward_mean = np.mean(rewards_possible)
reward_stdev = np.std(rewards_possible)

def normalise_reward(reward, prev_lives, curr_lives):
    """
    Given a reward from the game normalises it zero mean and unit variance.
    
    :param reward: the reward returned by the environment for taking the step.  In space invaders this is the
                   score after the action minus the score before the action.
    :param prev_lives: the lives before the action was taken.
    :param curr_lives: the lives after the action was taken.

    """
    # Check if we've lost a life and set the reward accordingly
    if prev_lives != curr_lives:
        reward = -305

    return (reward - reward_mean) / reward_stdev


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