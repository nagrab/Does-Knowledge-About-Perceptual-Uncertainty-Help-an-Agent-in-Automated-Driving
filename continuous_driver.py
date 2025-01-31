import os
import sys
import time
import random
import numpy as np
import argparse
import logging
import pickle
import torch
from distutils.util import strtobool
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from encoder_init import EncodeState
from networks.on_policy.ppo.agent import PPOAgent
from simulation.connection import ClientConnection
from simulation.environment import CarlaEnvironment
from parameters import *
import csv

'''
Source: https://github.com/idreesshaikh/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning
'''
print('Start continous driver')

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='ppo', help='name of the experiment')
    parser.add_argument('--env-name', type=str, default='carla', help='name of the simulation environment')
    parser.add_argument('--learning-rate', type=float, default=PPO_LEARNING_RATE, help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=SEED, help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=TOTAL_TIMESTEPS, help='total timesteps of the experiment')
    parser.add_argument('--action-std-init', type=float, default=ACTION_STD_INIT, help='initial exploration noise')
    parser.add_argument('--test-timesteps', type=int, default=TEST_TIMESTEPS, help='timesteps to test our model')
    parser.add_argument('--episode-length', type=int, default=EPISODE_LENGTH, help='max timesteps in an episode')
    parser.add_argument('--train', default=False, type=boolean_string, help='is it training?')
    parser.add_argument('--town', type=str, default="Town02", help='which town do you like?')
    parser.add_argument('--load-checkpoint', type=bool, default=MODEL_LOAD, help='resume training?')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, cuda will not be enabled by deafult')
    parser.add_argument('--done', type=int, default=599, help='for validation episode number')
    parser.add_argument('--case', type=str, default="e", help='Which case for validation/training as the state space do you want?')
    parser.add_argument('--sampler', type=bool, default = False, help ='Should the cases be sampled with time steps?')
    parser.add_argument('--timesampler', type=str, default="7500", help='On which time should be sampled. If random-> range will be random sampled')
    parser.add_argument('--run_list', type=bool, default = False, help='Do you already had a run in this setup?')
    parser.add_argument('--runpath', type=str, default="", help = 'Where are you first run?')
    args = parser.parse_args()
    print(args.train)
    return args

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

'''
Cases:
a: Front vehicle is disappeared        +eg-+
b: Vehicle in the back is disappeared -eg++
c: All front vehicles are disappeard +eg--
d: All vehicles are disappeared  -eg--
e: No vehicle is disappeared    +eg++   
'''

def runner():

    #========================================================================
    #                           BASIC PARAMETER & LOGGING SETUP
    #========================================================================
    
    args = parse_args()
    exp_name = args.exp_name
    train = args.train
    town = args.town
    checkpoint_load = args.load_checkpoint
    total_timesteps = args.total_timesteps
    action_std_init = args.action_std_init
    case_tmax = args.case
    names_cases = args.case
    sampler = args.sampler
    timesampler = args.timesampler
    try:
        if exp_name == 'ppo':
            run_name = "PPO"
        else:
            """
            
            Here the functionality can be extended to different algorithms.

            """ 
            sys.exit() 
    except Exception as e:
        print(e.message)
        sys.exit()
    
    # if train == True:
    #     writer = SummaryWriter(f"runs/{run_name}_{action_std_init}_2000000/{town}_{SAVE_NAME}")
    # else:
    #     writer = SummaryWriter(f"runs/{run_name}_{action_std_init}_2000000_TEST/{town}_{SAVE_NAME}_case_{names_cases}_{TNAME}_{RUN_NUMBER}")
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()])))


    #Seeding to reproduce the results 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    
    action_std_decay_rate = ACTION_STD_DECAY_RATE#0.05
    min_action_std = MIN_ACTION_STD#0.05   
    action_std_decay_freq = ACTION_STD_DECAY_FREQ #5e5
    timestep = 0
    episode = 0
    cumulative_score = 0
    episodic_length = list()
    scores = list()
    deviation_from_center = 0
    distance_covered = 0
    sample_list = []
    time_sampler_list = []
    if sampler == True:
        if "a" in case_tmax:
            sample_list.append('a')
        if "b" in case_tmax:
            sample_list.append('b')
        if "c" in case_tmax:
            sample_list.append('c')
        if "d" in case_tmax:
            sample_list.append('d')
        if "e" in case_tmax:
            sample_list.append('e')
    if timesampler == "random":
        time_sampler_list = [20,50,100,150,200,400]
    else:
        time_sampler_list = [int(timesampler)]
    
    
    
    #========================================================================
    #                           CREATING THE SIMULATION
    #========================================================================

    try:
        client, world, carla_process = ClientConnection(town).setup()
        logging.info("Connection has been setup successfully.")
    except:
        logging.error("Connection has been refused by the server.")
        ConnectionRefusedError
    if train:
        env = CarlaEnvironment(client, world,town)
    else:
        env = CarlaEnvironment(client, world,town, checkpoint_frequency=None)
    encode = EncodeState(LATENT_DIM)
    
    #print('encode')
    location_list = []

    #========================================================================
    #                           ALGORITHM
    #========================================================================
    # done_list = [0,99,199,299,399,499,599,699,799,899]
    # csvfile = '/home/carla-admin/ma-grabowsky/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning/vae_with_shifting/results/Town02_train_wo_pert_and_unc_lprev_lcurr_times_1_t_tmax_min_d_mal_50_new_br_th_train_each_eps_tim_mal_ga_0.999_va_0.1_500000.0_0.025_0.025_nw_img_nw_inertia_0_2_7500_1e-05/reward.csv' 
    # with open(csvfile, newline='') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=',')
    #     print(csvfile)
    #     for row in reader:
    #         if row[0] == 'episode':
    #             continue
    #         done_list.append(int(row[0])-1)

    # done_list = np.unique(done_list)
    # print(f'This episodes are validated: {done_list}')
    
    try:
        time.sleep(0.5)
        
        if checkpoint_load:
            # print('load checkpoint')
            # print(SAVE_NAME)
            # #print(len(next(os.walk(f'checkpoints/PPO/{town}_{SAVE_NAME}'))[2]))# - 1)
            chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}_{SAVE_NAME}'))[2]) - 1
            chkpt_file = f'checkpoints/PPO/{town}_{SAVE_NAME}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
            print('load checkpoint')
            print(SAVE_NAME)
            chkpt_file_weights = None
            # print(len(next(os.walk(f'checkpoints/PPO/{town}_{SAVE_NAME}'))[2]))# - 1)
            # chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}_{SAVE_NAME}'))[2]) - 1
            # chkpt_file = 'checkpoints/PPO/Town02_train_wo_pert_and_unc_lprev_lcurr_times_1_t_tmax_min_d_mal_50_new_br_th_train_each_eps_tim_mal_ga_0.999_va_0.1_500000.0_0.025_0.025_nw_img_nw_inertia_3_2_7500_1e-05/checkpoint_ppo_599.pickle'
            # chkpt_file_weights = '/home/carla-admin/ma-grabowsky/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning/vae_with_shifting/ppo/Town02_train_wo_pert_and_unc_lprev_lcurr_times_1_t_tmax_min_d_mal_50_new_br_th_train_each_eps_tim_mal_ga_0.999_va_0.1_500000.0_0.025_0.025_nw_img_nw_inertia_3_2_7500_1e-05/ppo_policy_599_.pth'
            print(chkpt_file)
            with open(chkpt_file, 'rb') as f:
                data = pickle.load(f)
                episode = data['episode']
                timestep = data['timestep']
                cumulative_score = data['cumulative_score']
                action_std_init = data['action_std_init']
            print('start loading agent')
            agent = PPOAgent(town, action_std_init)
            agent.load(chkpt_file_weights)
            print('Agent load')
            
        else:
            if train == False:
                
                print('load checkpoint')
                print(SAVE_NAME)
                #print(len(next(os.walk(f'checkpoints/PPO/{town}_{SAVE_NAME}'))[2]))# - 1)
                #chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}_{SAVE_NAME}'))[2]) - 1
                chkpt_file = f'/home/carla-admin/ma-grabowsky/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning/checkpoints/PPO/{town}_{SAVE_NAME}/checkpoint_ppo_'+str(args.done)+'.pickle'
                chkpt_file_weights = f'/home/carla-admin/ma-grabowsky/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning/vae_with_shifting/ppo/{town}_{SAVE_NAME}/ppo_policy_{args.done}_.pth'
                print(chkpt_file)
                print(chkpt_file_weights)
                with open(chkpt_file, 'rb') as f:
                    data = pickle.load(f)
                    episode = data['episode']
                    timestep = data['timestep']
                    cumulative_score = data['cumulative_score']
                    action_std_init = data['action_std_init']
                print(f'Validation episode: {episode}')
                print('Agent is loading')
                agent = PPOAgent(town, action_std_init)
                print('Agent starts loading')
                agent.load(chkpt_file_weights)
                print('Agent loaded')
                for params in agent.old_policy.actor.parameters():
                    params.requires_grad = False
            else:
                agent = PPOAgent(town, action_std_init)
        if train:
            #Training
            #print('training')
            print(f'Training for {total_timesteps}')
            while timestep < total_timesteps:
                run = []
                #print(f'timestep: {timestep}')
                if sampler == True:
                    case_tmax = random.choice(sample_list)
                switching_time = random.choice(time_sampler_list)
                run.append((case_tmax,switching_time))
                if args.run_list == True:
                    print(f'{args.runpath}'+f'/{episode}'+f'/run_{names_cases}_{episode}.txt')
                    file = open((f'{args.runpath}'+f'/{episode}'+f'/run_{names_cases}_{episode}.txt'),'r')
                    run1_list = eval(file.read())
                    case_tmax = str(run1_list[0][0])
                    switching_time = int(run1_list[0][1])
                    run1_list.pop(0)
                    run = []
                    print(case_tmax)
                print(switching_time)
                observation = env.reset(episode, False, case = case_tmax,case_name=names_cases)
                #print(f'Spawning case: {sp_case}')
                observation = encode.process(observation,timestep)
                current_ep_reward = 0
                t1 = datetime.now()
                start = time.time()
                writing_time = []
                #print(np.shape(observation))
                
                for t in range(args.episode_length):
                    if (timestep+1)%switching_time == 0:
                        if args.run_list == True and len(run1_list) !=0:
                            case_tmax = str(run1_list[0][0])
                            switching_time = int(run1_list[0][1])
                            run1_list.pop(0)
                        else:
                            case_tmax = random.choice(sample_list)
                            switching_time = random.choice(time_sampler_list)
                            run.append((case_tmax,switching_time))
                    #print(f'observation: {observation}')
                    # select action with policy
                    action = agent.get_action(observation, train=True)
                    observation, reward, done, info , loc ,velocity, action_time, brake, distance_wp_ego, throttle, action_index, lprev_lcurr,t_tmax , d_min_dmax,dist_ego_vehi_priv, loc_prev_veh, vel_front, finished_route = env.step(action,episode, val = False,case = case_tmax,case_name=names_cases)#, log_wp, loc_ego, wp_index, wp5index, wp75index, wp_idx= env.step(action)
                    end = time.time()
                    row= [[loc.x, loc.y, velocity, end-start, reward, action_time, brake, throttle, action_index, lprev_lcurr, t_tmax,d_min_dmax,dist_ego_vehi_priv, loc_prev_veh.x, loc_prev_veh.y, action_std_init, vel_front, case_tmax]]#, distance_wp_ego, log_wp, loc_ego, wp_index, wp5index, wp75index, wp_idx]]
                    fields = ['x','y', 'velocity', 'time [s]', 'reward', 'action_time', 'brake', 'throttle', 'action_idx', 'distance_prev_curr', 'time_reward', 'distance_total','distance_ego_vehicle_prev', 'x_prev_vehic', 'y_prev_vehic', 'variance_action_std', 'velocity_front','case']#, 'distance_wp_ego', 'location_wp_r25', 'location_ego', 'wp25_index', 'wp5index', 'wp75index', 'wp_idx']
                    if not os.path.exists(os.path.join(SAVE_PATH, f'{town}_{SAVE_NAME}')):
                        os.makedirs(os.path.join(SAVE_PATH,f'{town}_{SAVE_NAME}'))
                    start_writing = time.time()
                    with open(os.path.join(SAVE_PATH,f'{town}_{SAVE_NAME}', f'episode_{episode+1}.csv'), 'a') as file:
                        csvwriter = csv.writer(file)
                        if not os.path.exists(os.path.join(SAVE_PATH,f'{town}_{SAVE_NAME}', f'episode_{episode+1}.csv')) or os.stat(os.path.join(SAVE_PATH,f'{town}_{SAVE_NAME}', f'episode_{episode+1}.csv')).st_size == 0:
                            csvwriter.writerow(fields)
                        csvwriter.writerows(row)
                    end_writing = time.time()
                    writing_time.append(end_writing-start_writing)
                    if observation is None:
                        print('oberservation is none')
                        break
                    observation = encode.process(observation,timestep)
                    # print(reward)
                    agent.memory.rewards.append(reward)
                    #print(len(agent.memory.rewards))
                    agent.memory.dones.append(done)
                    
                    timestep +=1
                    current_ep_reward += reward
                    
                    if timestep % action_std_decay_freq == 0:
                        action_std_init =  agent.decay_action_std(action_std_decay_rate, min_action_std)
                        print(action_std_init)
                    if timestep == total_timesteps -1:
                        agent.chkpt_save()

                    # break; if the episode is over
                    if done:
                        episode += 1

                        t2 = datetime.now()
                        t3 = t2-t1
                        
                        episodic_length.append(abs(t3.total_seconds()))
                        break   
                #env.string(loc, episode)
                #print('episode is done')
                deviation_from_center += info[1]
                distance_covered += info[0]
                
                scores.append(current_ep_reward)
                
                if checkpoint_load:
                    cumulative_score = ((cumulative_score * (episode - 1)) + current_ep_reward) / (episode)
                else:
                    cumulative_score = np.mean(scores)


                print('Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score))
                if episode % 1 == 0:
                    agent.learn()
                    agent.chkpt_save()
                    if not os.path.exists(f'checkpoints/PPO/{town}_{SAVE_NAME}'):
                        os.makedirs(f'checkpoints/PPO/{town}_{SAVE_NAME}')
                        print('Checkpoint dir is created')
                    chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}_{SAVE_NAME}'))[2])
                    if chkt_file_nums != 0:
                        chkt_file_nums -=1
                    chkpt_file = f'checkpoints/PPO/{town}_{SAVE_NAME}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
                    data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep, 'action_std_init': action_std_init}
                    with open(chkpt_file, 'wb') as handle:
                        pickle.dump(data_obj, handle)
                    
                
                if episode % 1 == 0:
                    # writer.add_scalar("Episodic Reward/episode", scores[-1], episode)
                    # writer.add_scalar("Cumulative Reward/info", cumulative_score, episode)
                    # writer.add_scalar("Cumulative Reward/(t)", cumulative_score, timestep)
                    # writer.add_scalar("Average Episodic Reward/info", np.mean(scores[-1]), episode)
                    # writer.add_scalar("Average Reward/(t)", np.mean(scores[-1]), timestep)
                    # writer.add_scalar("Episode Length (s)/info", np.mean(episodic_length), episode)
                    # writer.add_scalar("Reward/(t)", current_ep_reward, timestep)
                    # writer.add_scalar("Average Deviation from Center/episode", deviation_from_center/1, episode)
                    # writer.add_scalar("Average Deviation from Center/(t)", deviation_from_center/1, timestep)
                    # writer.add_scalar("Average Distance Covered (m)/episode", distance_covered/1, episode)
                    # writer.add_scalar("Average Distance Covered (m)/(t)", distance_covered/1, timestep)
                    episodic_length = list()
                    deviation_from_center = 0
                    distance_covered = 0

                if episode % 1 == 0:
                    agent.save()
                    chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}_{SAVE_NAME}'))[2])
                    chkpt_file = f'checkpoints/PPO/{town}_{SAVE_NAME}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
                    data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep, 'action_std_init': action_std_init}
                    with open(chkpt_file, 'wb') as handle:
                        pickle.dump(data_obj, handle)
            #env.save_location(location_list)
                if episode % 100 == 0:
                    curr_epis = 0
                    timestep_val = 0
                    scores_valid = []
                    print('Start Validation')
                    for params in agent.old_policy.actor.parameters():
                        params.requires_grad = False
                    while curr_epis < args.test_timesteps:
                        print(f'Validation episode: {curr_epis}')
                        print(sample_list)
                        if sampler == True:
                            case_tmax = random.choice(sample_list)
                        switching_time = random.choice(time_sampler_list)
                        run.append((case_tmax,switching_time))
                        observation = env.reset(episode*100+curr_epis, True,case = case_tmax,case_name=names_cases)
                 
                        observation = encode.process(observation, timestep_val)

                        current_ep_reward = 0
                        t1 = datetime.now()
                        start = time.time()
                        for t in range(args.episode_length):
                            # select action with policy
                            if (timestep_val+1)%switching_time == 0:
                                if args.run_list == True and len(run1_list) !=0:
                                    case_tmax = str(run1_list[0][0])
                                    switching_time = int(run1_list[0][1])
                                    run1_list.pop(0)
                                else:
                                    case_tmax = random.choice(sample_list)
                                    switching_time = random.choice(time_sampler_list)
                                    run.append((case_tmax,switching_time))
                            action = agent.get_action(observation, train=False)
                            observation, reward, done, info , loc ,velocity, action_time, brake, distance_wp_ego, throttle, action_index, lprev_lcurr,t_tmax , d_min_dmax,dist_ego_vehi_priv, loc_prev_veh, vel_front, finished_route = env.step(action,episode*100+curr_epis, True,case = case_tmax,case_name=names_cases)#, log_wp, loc_ego, wp_index, wp5index, wp75index, wp_idx= env.step(action)
                            end = time.time()
                            row= [[loc.x, loc.y, velocity, end-start, reward, action_time, brake, throttle, action_index, lprev_lcurr, t_tmax,d_min_dmax,dist_ego_vehi_priv, loc_prev_veh.x, loc_prev_veh.y, action_std_init, vel_front,case_tmax]]#, distance_wp_ego, log_wp, loc_ego, wp_index, wp5index, wp75index, wp_idx]]
                            fields = ['x','y', 'velocity', 'time [s]', 'reward', 'action_time', 'brake', 'throttle', 'action_idx', 'distance_prev_curr', 'time_reward', 'distance_total','distance_ego_vehicle_prev', 'x_prev_vehic', 'y_prev_vehic', 'variance_action_std', 'velocity_front','case']#, 'distance_wp_ego', 'location_wp_r25', 'location_ego', 'wp25_index', 'wp5index', 'wp75index', 'wp_idx']
                            if not os.path.exists(os.path.join(SAVE_PATH,f'{town}_{SAVE_NAME}_validation',f'{episode*100+curr_epis+1}')):
                                os.makedirs(os.path.join(SAVE_PATH,f'{town}_{SAVE_NAME}_validation',f'{episode*100+curr_epis+1}'))
                            with open(os.path.join(SAVE_PATH,f'{town}_{SAVE_NAME}_validation',f'{episode*100+curr_epis+1}', f'validation_{episode*100+curr_epis+1}.csv'), 'a') as file:
                                csvwriter = csv.writer(file)
                                if not os.path.exists(os.path.join(SAVE_PATH,f'{town}_{SAVE_NAME}_validation',f'{episode*100+curr_epis+1}', f'validation_{episode*100+curr_epis+1}.csv')) or os.stat(os.path.join(SAVE_PATH,f'{town}_{SAVE_NAME}_validation', f'{episode*100+curr_epis+1}',f'validation_{episode*100+curr_epis+1}.csv')).st_size == 0:
                                    csvwriter.writerow(fields)
                                csvwriter.writerows(row)
                        
                            if observation is None:
                                break
                            observation = encode.process(observation, timestep_val)
                            
                            timestep_val +=1
                            current_ep_reward += reward
                            # break; if the episode is over
                            if done:
                                #episode += 1

                                t2 = datetime.now()
                                t3 = t2-t1
                                
                                episodic_length.append(abs(t3.total_seconds()))
                                break
                        deviation_from_center += info[1]
                        distance_covered += info[0]
                        
                        scores_valid.append(current_ep_reward)
                        cumulative_score_valid = np.mean(scores_valid)

                        print('Validation: Episode: {}'.format(episode*100+curr_epis+1),', Timestep: {}'.format(timestep_val),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score_valid))
                        
                        # writer.add_scalar("Validation: Episodic Reward/episode", scores_valid[-1], episode*100+curr_epis+1)
                        # writer.add_scalar("Validation: Cumulative Reward/info", cumulative_score_valid, episode*100+curr_epis+1)
                        # writer.add_scalar("Validation: Cumulative Reward/(t)", cumulative_score_valid, episode*100+curr_epis+1)
                        # writer.add_scalar("Validation: Episode Length (s)/info", np.mean(episodic_length), episode*100+curr_epis+1)
                        # writer.add_scalar("Validation: Reward/(t)", current_ep_reward, timestep_val)
                        # writer.add_scalar("Validation: Deviation from Center/episode", deviation_from_center, episode*100+curr_epis+1)
                        # writer.add_scalar("Validation: Deviation from Center/(t)", deviation_from_center, episode*100+curr_epis+1)
                        # writer.add_scalar("Validation: Distance Covered (m)/episode", distance_covered, episode*100+curr_epis+1)
                        # writer.add_scalar("Validation: Distance Covered (m)/(t)", distance_covered, episode*100+curr_epis+1)

                        episodic_length = list()
                        deviation_from_center = 0
                        distance_covered = 0
                        curr_epis += 1
                    print('Training reactivation')
                    for params in agent.old_policy.actor.parameters():
                        params.requires_grad = True
            print("Terminating the run.")
            sys.exit()
        else:
            print('Start testing')
            #Testing
            curr_epis = 0
            timestep_val = 0
            
            while curr_epis < args.test_timesteps:
                run = []
                print(sample_list)
                if sampler == True:
                    case_tmax = random.choice(sample_list)
                switching_time = random.choice(time_sampler_list)
                run.append((case_tmax,switching_time))
                if args.run_list == True:
                    print(f'{args.runpath}'+f'/{episode*100+curr_epis+1}'+f'/run_{names_cases}_{episode*100+curr_epis+1}.txt')
                    file = open((f'{args.runpath}'+f'/{episode*100+curr_epis+1}'+f'/run_{names_cases}_{episode*100+curr_epis+1}.txt'),'r')
                    run1_list = eval(file.read())
                    case_tmax = str(run1_list[0][0])
                    switching_time = int(run1_list[0][1])
                    run1_list.pop(0)
                    run = []
                    print(case_tmax)
                print(switching_time)
                observation = env.reset(episode*100+curr_epis, True, case = case_tmax,case_name=names_cases)
                observation = encode.process(observation, timestep = timestep_val)

                current_ep_reward = 0
                t1 = datetime.now()
                start = time.time()
                
                for t in range(args.episode_length):
                    if (timestep_val+1)%switching_time == 0:
                        if args.run_list == True and len(run1_list) !=0:
                            case_tmax = str(run1_list[0][0])
                            switching_time = int(run1_list[0][1])
                            run1_list.pop(0)
                        else:
                            case_tmax = random.choice(sample_list)
                            switching_time = random.choice(time_sampler_list)
                            run.append((case_tmax,switching_time))
                        #print(f"Change for {switching_time} the case into {case_tmax}")
                    # select action with policy
                    action = agent.get_action(observation, train=False)
                    observation, reward, done, info , loc ,velocity, action_time, brake, distance_wp_ego, throttle, action_index, lprev_lcurr,t_tmax , d_min_dmax,dist_ego_vehi_priv, loc_prev_veh, vel_front, finished_route = env.step(action,episode*100+curr_epis, True, case = case_tmax,case_name=names_cases)
                    end = time.time()
                    row= [[loc.x, loc.y, velocity, end-start, reward, action_time, brake, throttle, action_index, lprev_lcurr, t_tmax,d_min_dmax,dist_ego_vehi_priv, loc_prev_veh.x, loc_prev_veh.y, action_std_init, vel_front, case_tmax, switching_time]]#, distance_wp_ego, log_wp, loc_ego, wp_index, wp5index, wp75index, wp_idx]]
                    fields = ['x','y', 'velocity', 'time [s]', 'reward', 'action_time', 'brake', 'throttle', 'action_idx', 'distance_prev_curr', 'time_reward', 'distance_total','distance_ego_vehicle_prev', 'x_prev_vehic', 'y_prev_vehic', 'variance_action_std', 'velocity_front','case','sample time']#, 'distance_wp_ego', 'location_wp_r25', 'location_ego', 'wp25_index', 'wp5index', 'wp75index', 'wp_idx']
                    if not os.path.exists(os.path.join(f'{SAVE_PATH}',f'{town}_{SAVE_NAME}_validation_{names_cases}_{TNAME}_{RUN_NUMBER}',f'{episode*100+curr_epis+1}')):
                        os.makedirs(os.path.join(f'{SAVE_PATH}',f'{town}_{SAVE_NAME}_validation_{names_cases}_{TNAME}_{RUN_NUMBER}',f'{episode*100+curr_epis+1}'))
                    with open(os.path.join(f'{SAVE_PATH}',f'{town}_{SAVE_NAME}_validation_{names_cases}_{TNAME}_{RUN_NUMBER}',f'{episode*100+curr_epis+1}', f'validation_{episode*100+curr_epis+1}.csv'), 'a') as file:
                        csvwriter = csv.writer(file)
                        if not os.path.exists(os.path.join(f'{SAVE_PATH}',f'{town}_{SAVE_NAME}_validation_{names_cases}_{TNAME}_{RUN_NUMBER}',f'{episode*100+curr_epis+1}', f'validation_{episode*100+curr_epis+1}.csv')) or os.stat(os.path.join(f'{SAVE_PATH}',f'{town}_{SAVE_NAME}_validation_{names_cases}_{TNAME}_{RUN_NUMBER}', f'{episode*100+curr_epis+1}',f'validation_{episode*100+curr_epis+1}.csv')).st_size == 0:
                            csvwriter.writerow(fields)
                        csvwriter.writerows(row)
                   
                    if observation is None:
                        break
                    observation = encode.process(observation, timestep = timestep_val)
                    
                    timestep_val +=1
                    current_ep_reward += reward
                    # break; if the episode is over
                    if done:
                        #episode += 1

                        t2 = datetime.now()
                        t3 = t2-t1
                        
                        episodic_length.append(abs(t3.total_seconds()))
                        break
                deviation_from_center += info[1]
                distance_covered += info[0]
                
                scores.append(current_ep_reward)
                cumulative_score = np.mean(scores)
                print('Episode ends')
                print('Validation {} : Episode: {}'.format(case_tmax,episode*100+curr_epis+1),', Timestep: {}'.format(timestep_val),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score))
                
                # writer.add_scalar("TEST: Episodic Reward/episode", scores[-1], episode)
                # writer.add_scalar("TEST: Cumulative Reward/info", cumulative_score, episode)
                # writer.add_scalar("TEST: Cumulative Reward/(t)", cumulative_score, timestep)
                # writer.add_scalar("TEST: Episode Length (s)/info", np.mean(episodic_length), episode)
                # writer.add_scalar("TEST: Reward/(t)", current_ep_reward, timestep)
                # writer.add_scalar("TEST: Deviation from Center/episode", deviation_from_center, episode)
                # writer.add_scalar("TEST: Deviation from Center/(t)", deviation_from_center, timestep)
                # writer.add_scalar("TEST: Distance Covered (m)/episode", distance_covered, episode)
                # writer.add_scalar("TEST: Distance Covered (m)/(t)", distance_covered, timestep)
                if len(run)!=0:
                    with open(f'{SAVE_PATH}'+'/'+f'{town}_{SAVE_NAME}_validation_{names_cases}_{TNAME}_{RUN_NUMBER}'+f'/{episode*100+curr_epis+1}'+f'/run_{names_cases}_{episode*100+curr_epis+1}.txt','w') as file:
                        file.write(str(run))
                        file.close()
                episodic_length = list()
                deviation_from_center = 0
                distance_covered = 0
                curr_epis+=1
            print("Terminating the run.")
            sys.exit()
            carla_process.kill()
            

    finally:
        sys.exit()
        carla_process.kill()
       


if __name__ == "__main__":
    try:        
        runner()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        
        print('\nExit')
