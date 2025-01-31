"""

    All the much needed hyper-parameters needed for the algorithm implementation. 
    Source: https://github.com/idreesshaikh/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning

"""

MODEL_LOAD = False
SEED = 0
BATCH_SIZE = 1
GAMMA = 0.999
MEMORY_SIZE = 5000
EPISODES = 1000

#VAE Bottleneck
LATENT_DIM = 95

#Proximal Policy Optimization (hyper)parameters
EPISODE_LENGTH = 7500
TOTAL_TIMESTEPS = 2e6
ACTION_STD_INIT = 0.1
ACTION_STD_DECAY_FREQ = 5e5
ACTION_STD_DECAY_RATE = 0.025
MIN_ACTION_STD = 0.025
TEST_TIMESTEPS = 20
PPO_LEARNING_RATE = 1e-5 
PPO_CHECKPOINT_DIR = 'path/to/checkpoint/ppo/'
POLICY_CLIP = 0.2
BETA=3
BETA_TILDE=2
TMAX=7500
TNAME='fully'
RUN_NUMBER=3
#Savepath
SAVE_PATH = ''
SAVE_NAME = ''
