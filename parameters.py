"""

    All the much needed hyper-parameters needed for the algorithm implementation. 
    Source: https://github.com/idreesshaikh/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning

"""

MODEL_LOAD = False
SEED = 0
BATCH_SIZE = 1
# IM_WIDTH = 160
# IM_HEIGHT = 80
GAMMA = 0.999
MEMORY_SIZE = 5000
EPISODES = 1000

#VAE Bottleneck
LATENT_DIM = 95

#Dueling DQN (hyper)parameters
DQN_LEARNING_RATE = 0.0001
EPSILON = 1.00
EPSILON_END = 0.05
EPSILON_DECREMENT = 0.00001

REPLACE_NETWORK = 5
DQN_CHECKPOINT_DIR = 'preTrained_models/ddqn'
MODEL_ONLINE = 'carla_dueling_dqn_online.pth'
MODEL_TARGET = 'carla_dueling_dqn_target.pth'


#Proximal Policy Optimization (hyper)parameters
EPISODE_LENGTH = 7500
TOTAL_TIMESTEPS = 2e6#4e6#6e6#2e6#7e5 nur Testzwecke
ACTION_STD_INIT = 0.1#0.1 #0.2
ACTION_STD_DECAY_FREQ = 5e5#2e5 #5e5(oracle)
ACTION_STD_DECAY_RATE = 0.025#5#0.025 #0.05
MIN_ACTION_STD = 0.025#5#0.025  #0.05
TEST_TIMESTEPS = 20#5e4
PPO_LEARNING_RATE = 1e-5  #1e-4
PPO_CHECKPOINT_DIR = 'vae_with_shifting/ppo/'
POLICY_CLIP = 0.2
BETA=3
BETA_TILDE=2
TMAX=7500
TNAME='fully'
RUN_NUMBER=3
#Savepath
SAVE_NAME = f"train_wo_pert_and_unc_lprev_lcurr_times_1_t_tmax_min_d_mal_50_new_br_th_train_each_eps_tim_mal_ga_{GAMMA}_va_{ACTION_STD_INIT}_{ACTION_STD_DECAY_FREQ}_{ACTION_STD_DECAY_RATE}_{MIN_ACTION_STD}_nw_img_nw_inertia_{BETA}_{BETA_TILDE}_{TMAX}_{PPO_LEARNING_RATE}"