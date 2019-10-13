# Installation

You will need a *nix based machine to run the code (i.e. Linux/Mac).

1. If running on AWS a r5.xlarge is recommended as you need a machine with 32GB of RAM (or be prepared to use SSD swap).  This is a GPU machine, modifications are required to the code to utilise a CPU.  The code takes approximately two days to run and an average score of around 700 is obtained.  Note use the barebone Ubuntu AMI and follow the instructions below. 
1. Install the Python 3 version of Ananconda: https://www.anaconda.com/distribution/.  For a command line installation follow https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart.
1. Tensorflow and Keras must be additionally installed: ```conda install -c anaconda tensorflow``` and ```conda install -c anaconda keras``` or alternatively use the Anaconda UI package manager.  If running on AWS Tensorflow and Keras must be installed through ```pip``` instead as the conda versions have a memory leak.
1. Follow the instructions at https://github.com/openai/gym#installation to install OpenAI Gym with the Atari environment; in short `pip install gym` followed by `pip install gym[atari]`.
1. Edit the `game_selection.py` file to select the game to play; space invaders or pong.
1. Train an agent by running `python train.py`.
1. A training graph can be plotted by modifying the `plot_training_graph.py` file to change the `training_to_plot` variable to the timestamp of the training results file in the `results` directory then executing `python plot_training_graph.py`.  The graph will be saved in the `results` directory.
1. Generate videos in the `videos` directory of the trained agent by executing `python run_trained_agent.py`.
