import numpy as np

def normalise_reward(reward, prev_lives, curr_lives, game_over):
    """
    No reward normalisation is required in pong; -1 indicates opponent has
    scored, 0 indicates game is still in play and 1 indicates player has
    scored.
    """
    return reward


def get_num_actions():
    """
    Returns the number of actions available in the game; up, down, do nothing.
    """
    return 3


def get_gym_game_name():
    """
    Get the name of the game as used by OpenAI gym.
    """
    return 'PongNoFrameskip-v4'


def crop_frame(frame):
    """
    Given a game frame, crops the image so only the playable area is in view.
    """
    return frame[33:195,:]


def choose_random_action(env):
    """
    Given the environment, chooses a random action for the agent.
    """
    # Ignore the environment as it has several no-op actions, simply choose
    # from do nothing, up or down.
    return np.random.randint(0, 3)


def convert_action_for_env(action):
    """
    We work with indexes of 0 for no action, 1 for up and 2 for down.
    Environment expects 1 for no action, 2 for up and 3 for down.
    """
    return action+1


def get_learning_rate():
    """
    Get the learning rate for gradient descent.
    """
    return 0.00025