# Installation

You will need a *nix based machine to run the code (i.e. Linux/Mac).

1. If you want a high score on Space Invaders, please use the the https://github.com/dscook/atari-with-ai/tree/space-invaders branch.
1. Install the Python 3 version of Ananconda: https://www.anaconda.com/distribution/.  For a command line installation follow https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart.
1. Tensorflow and Keras must be additionally installed: ```conda install -c anaconda tensorflow``` and ```conda install -c anaconda keras``` or alternatively use the Anaconda UI package manager.  If running on AWS (this code currently runs on CPU only, starting from a barebone Ubuntu AMI is recommended) Tensorflow and Keras must be installed through ```pip``` instead as the conda versions have a memory leak.
1. Follow the instructions at https://github.com/openai/gym#installation to install OpenAI Gym with the Atari environment; in short `pip install gym` followed by `pip install gym[atari]`.
1. Edit the `game_selection.py` file to select the game to play; space invaders or pong.
1. Train an agent by running `python train.py`.
1. A training graph can be plotted by modifying the `plot_training_graph.py` file to change the `training_to_plot` variable to the timestamp of the training results file in the `results` directory then executing `python plot_training_graph.py`.  The graph will be saved in the `results` directory.
1. Generate videos in the `videos` directory of the trained agent by executing `python run_trained_agent.py`.

# Acknowledgements

The code is an implementation of the following papers:

* Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D. and Riedmiller, M., 2013. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.
* Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A.,
Riedmiller, M., Fidjeland, A.K., Ostrovski, G. et al., 2015. Human-level control through deep
reinforcement learning. Nature Publishing Group, vol. 518, p.529.
