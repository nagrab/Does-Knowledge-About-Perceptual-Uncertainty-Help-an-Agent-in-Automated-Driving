import matplotlib.pyplot as plt
import numpy as np
import os 
import csv
import cv2
'''
Written by Natalie Grabowsky
This script is written to plot results of the training
'''

def plot_scatter(x,y, save_path, title, x_label, y_label, town_map = None, z = None):
    '''
    Plot scatterplots
    Inputparameter:
    -----------------------
    x        : List: List for x values
    y        : List: List for y value
    save_path:  Str: Path where the plot should be saved
    title    :  Str: Titel of plot
    x_label  :  Str: Label for x axis
    y_label  :  Str: Label for y axis
    town_map :  Str: Path for town 
    '''
    if town_map is not None:
        town = cv2.imread(town_map, cv2.COLOR_BGR2RGB)
        plt.imshow(town)

    try:
        plt.pcolor(z)
    except:
        pass
    plt.scatter(x,y,alpha=0.2,c=z,cmap='viridis', vmin=-0.1)#, vmin=0.0, vmax=25.0)#vmin=-0.1)#, vmin=0.0, vmax=25.0)
    cbar = plt.colorbar()
    cbar.set_label('reward')
    # plt.xlim(15,200)
    # plt.ylim(100,120)
    # plt.xlabel(x_label)
    # plt.ylabel(y_label)
    # plt.title(title.split('.')[0])
    # plt.savefig(os.path.join(save_path, title.split('.')[0]+'.png'))
    # plt.close()
    plt.figure(figsize=(25,5))
    plt.plot(z[:-1])
    plt.xlabel('time')
    plt.ylabel('reward')
    plt.title('reward')
    plt.savefig(os.path.join(save_path,'reward_over_time',title.split('.')[0]+'.png'))
    plt.close()

def plot_spawn_points(index, x, y, save_path):
    fig, ax = plt.subplots()
    scatter  = ax.scatter(x,y)
    for i in index:
        ax.annotate(str(i),(x[i],y[i]))
    plt.savefig(save_path)

def read_csv(csv_file, z=False):
    '''
    Imread csv file to split the data.
    Inputparameter:
    ----------------------------------
    csv_file:   Str: Csv file to imread

    Outputparameter:
    -----------------------------------
    x:        List: List of datas
    y:        List: List of datas
    '''
    x                   = []
    y                   = []
    vel                 = []
    time                = []
    reward              = []
    ac_time             = []
    brake               = []
    throttle            = []
    action              = []
    distance_prev_curr  = []
    time_rew             = []
    distance_total_curr = []
    distance_vehicle = []
    alpha = []
    x_front = []
    y_front = []
    vel_front = []
    with open(csv_file, newline='') as csvfile:
        print(csv_file)
        reader = csv.reader(csvfile, delimiter=',')

        for row in reader:
            if row[0]=='x':
                continue
            x.append(float(row[0]))
            y.append(float(row[1]))
            vel.append(float(row[2]))
            time.append(float(row[3]))
            reward.append(float(row[4]))
            ac_time.append(float(row[5]))
            brake.append(float(row[6]))
            throttle.append(float(row[7]))
            action.append(row[8])
            distance_prev_curr.append(float(row[9]))
            time_rew.append(float(row[10]))
            distance_total_curr.append(float(row[11]))
            distance_vehicle.append(float(row[12]))
            alpha.append(float(row[15]))
            x_front.append(float(row[13]))
            y_front.append(float(row[14]))
            vel_front.append(float(row[16]))
    return x,y, vel, time,reward,ac_time,brake,throttle,action,distance_prev_curr, time_rew,distance_total_curr, distance_vehicle , alpha, x_front, y_front, vel_front

def read_csv_spawn_points(csv_file):
    '''
    Read the csv file for spawn points.
    Inputparameter:
    ----------------------------------
    csv_file: Str: Csv file of spawn points

    Outputparameter:
    ----------------------------------
    Index: List: List of index
    x    : List: List of x values
    y    : List: List of y values
    '''
    index = []
    x     = []
    y     = []

    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        print(csv_file)
        for row in reader:
            try:
                if row[0]=='index':
                    continue
                index.append(int(row[0]))
                x.append(float(row[1]))
                y.append(float(row[4]))
            except:
                print(csv_file)

    return index, x, y

if __name__ == '__main__':
    # path = 'route_Town02'
    # x, y = read_csv(path, False)
    # plot_scatter(x, y, '', 'route_Town02_reward_waypoint_1','x', 'y')
    single = False
    total = True
    Beta = 3
    Beta_tilde = 2
    save_path = 'vae_with_shifting/results/Town02_train_wo_pert_and_unc_lprev_lcurr_times_1_t_tmax_min_d_mal_50_new_br_th_train_each_eps_tim_mal_ga_0.999_va_0.1_500000.0_0.025_0.025_nw_img_nw_inertia_3_2_7500_1e-05'
    source = 'Town02_train_wo_pert_and_unc_lprev_lcurr_times_1_t_tmax_min_d_mal_50_new_br_th_train_each_eps_tim_mal_ga_0.999_va_0.1_500000.0_0.025_0.025_nw_img_nw_inertia_3_2_7500_1e-05'
    path = '/home/carla-admin/ma-grabowsky/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning/reward_advantageTown02_train_wo_pert_and_unc_lprev_lcurr_times_1_t_tmax_min_d_mal_50_new_br_th_train_each_eps_tim_mal_ga_0.999_va_0.1_500000.0_0.025_0.025_nw_img_nw_inertia_3_2_7500_1e-05'
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
        print(save_path)
    if os.path.exists(os.path.join(save_path,'reward/'))==False:
        print('create directory for reward over time')
        os.makedirs(save_path+'/reward/')
    if os.path.exists(os.path.join(save_path,'brake_throttle/'))==False:
        print('create directory for brake throttle behavior')
        os.makedirs(save_path+'/brake_throttle/')
    if os.path.exists(os.path.join(save_path,'brake/'))==False:
        print('create directory for brake behavior')
        os.makedirs(save_path+'/brake/')
    if os.path.exists(os.path.join(save_path,'throttle/'))==False:
        print('create directory for throttle behavior')
        os.makedirs(save_path+'/throttle/')
    if os.path.exists(os.path.join(save_path,'brake_or_throttle/'))==False:
        print('create directory for brake or throttle behavior')
        os.makedirs(save_path+'/brake_or_throttle/')
    if os.path.exists(os.path.join(save_path,'brake_or_throttle_time/'))==False:
        print('create directory for brake or throttle over time behavior')
        os.makedirs(save_path+'/brake_or_throttle_time/')
    if os.path.exists(os.path.join(save_path,'reward_split/'))==False:
        print('create directory for reward split time behavior')
        os.makedirs(save_path+'/reward_split/')
    if os.path.exists(os.path.join(save_path,'discounted_reward_all/'))==False:
        print('create directory for reward split time behavior')
        os.makedirs(save_path+'/discounted_reward_all/')
    if os.path.exists(os.path.join(save_path, 'distribution_throttle_brake/'))== False:
        os.makedirs(save_path+'/distribution_throttle_brake/')
    if os.path.exists(os.path.join(save_path, 'velocity/')) == False:
        os.makedirs(save_path+'/velocity/')
    if os.path.exists(os.path.join(save_path, 'distance_front_vehicle/'))==False:
        os.makedirs(save_path+'/distance_front_vehicle/')
    if os.path.exists(os.path.join(save_path, 'combine_distances/'))==False:
        os.makedirs(save_path+'/combine_distances/')
    if os.path.exists(os.path.join(save_path, 'partial_reward/'))==False:
        os.makedirs(save_path+'/partial_reward/')
    # gamma = 0.90
    # if os.path.exists(os.path.join(save_path,f'discounted_reward_all_{gamma}/'))==False:
    #     print('create directory for reward split time behavior')
    #     os.makedirs(save_path+f'/discounted_reward_all_{gamma}/')
    for root,dir,files in os.walk(source):
        reward_total = []
        mean_reward = []
        std_reward = []
        brake_dist = []
        throttle_dist = []
        var_reward = []
        velocity_dist = []
        total_timesteps = 0
        episode_list = []
        timestep_list = []
        vel_list = []
        d_p_c = []
        d_v = []
        alpha_list = []
        for i in range(len(files)):
            # try:
            #     npy_file = np.load(path+'/'+file.split('_')[-1].split('.')[0]+'.npy', allow_pickle=True)
            # except:
            #     pass
            # dis_reward_total = npy_file[0][:]
    #town_map = 'Town02.jpg'
            title = f'episode_{i+1}'
            # if i+1 == 166:
            #     continue
            if single == True:
                if i+1 not in [3,13,36,45,48,89,96,97,134,135,136,148,152,153,166,197,200,205,216,218,249,250,253,254,256,257,258,259,260,261,262,273,275,281,289,291,292,293,300,301,304,305,315,317,320,322,323,342,347,349,351,352,353,361,366,370,372,374,376,378,379,385,386,388,389,390,392,393,397,402,405,407,408,410,423,425,434,494,504,505,506,516,518,519,520,521,523,524,525,526,533,536,537,538,544,545,550,553,556,557,559,561,562,565,568,569,570,572,574,575,576,584,586,587,589,595,596,599,600,602,606,608,609,612,615,616,618,620,621,627,630,637,638,643,645,646,648,650,651,655,657,659,660,661,665,666]:
                    continue
                # if os.path.exists(os.path.join(save_path,'reward_split',title.split('.')[0]+'.png')) == True:
                #     continue
            # if os.path.exists(os.path.join(save_path,f'reward_split',title.split('.')[0]+'.png')) and os.path.exists(os.path.join(save_path,f'reward',title.split('.')[0]+'.png')):
            #     continue
            brake_or_throttle = []
            #x,y,vel,time,reward,ac_time,brake,throttle,action,distance_prev_curr, distance_total_curr  = read_csv(root+'/'+f'episode_{i+1}.csv',True)
            x,y,vel,time,reward,ac_time,brake,throttle,action,distance_prev_curr,time_rew, distance_total_curr, distance_vehicle, alpha, x_front, y_front, vel_front  = read_csv(root+'/'+f'episode_{i+1}.csv',True)
            x_label = 'x'
            y_label = 'y'
            title = f'episode_{i+1}'
            reward_total.append(sum(reward))
            episode_list.append(i+1)
            timestep_list.append(len(reward))
            vel_list.append(max(distance_prev_curr))
            d_p_c.append(np.mean(distance_prev_curr))
            d_v.append(np.mean(distance_vehicle))
            mean_reward.append(np.mean(reward_total))
            std_reward.append(np.std(reward_total))
            var_reward.append(np.var(reward))
            velocity_dist.append(np.mean(vel))
            # for a in alpha:
            #     alpha_list.append(a)
            alpha_list.append(min(alpha))
            
            count_brake = 0
            count_throttle = 0
            for i in brake:
                if i > 0:
                    count_brake+=1
                else:
                    count_throttle+=1
            brake_dist.append(count_brake/len(brake))
            throttle_dist.append(count_throttle/len(throttle))
            total_timesteps += len(reward)
            for i in range(len(brake)):
                if brake[i]>throttle[i]:
                    brake_or_throttle.append(-brake[i])
                else:
                    brake_or_throttle.append(throttle[i])
            if single == True:
                beta_add = []
                beta_tilde_add = []
                for z, (x_f, y_f) in enumerate(zip(x_front,y_front)):
                    if x_f > 183 and y_f >110:
                        break
                distance_m = []
                for dis in distance_total_curr:
                    distance_m.append(150-300+150*dis)
                fig, ax1 = plt.subplots()
                color = 'tab:green'
                ax1.set_xlabel('time')
                ax1.set_ylabel('distance front vehicle',color=color)
                ax1.plot(distance_vehicle,color=color)
                ax2 = ax1.twinx()
                color='tab:blue'
                ax2.set_ylabel('distance endpoint',color=color)
                ax2.plot(distance_m,color=color)
                ax1.axvline(x = z, color='red', label='front vehicle drive on curve')
                fig.tight_layout()
                fig.savefig(os.path.join(save_path,'combine_distances',title.split('.')[0]+'_'+f'{round(sum(reward),2)}'+'.png'))
                plt.close()
                plt.rc('font', size=18) 
                plt.figure(figsize = (16,9))
                plt.plot(vel, label='velocity agent')
                plt.plot(vel_front, label='velocity front')
                plt.xlabel('time')
                plt.ylabel('velocity')
                plt.legend()
                #plt.title('velocity')
                plt.savefig(os.path.join(save_path,'velocity',title.split('.')[0]+'.png'))
                plt.close()
                for t in range(len(reward)):
                    beta_add.append(distance_prev_curr[t]*Beta)
                    beta_tilde_add.append(distance_prev_curr[t]*Beta_tilde*(1-t/7500))
                plt.rc('font', size=18)
                plt.figure(figsize=(16,9))
                plt.plot(beta_add, label = 'Beta')
                plt.plot(beta_tilde_add, label = 'Beta_tilde')
                plt.xlabel('time')
                plt.ylabel('partial reward')
                plt.legend()
                plt.savefig(os.path.join(save_path,'partial_reward',title.split('.')[0]+'.png'))
                plt.close()
                plt.rc('font', size=18) 
                plt.scatter(x,y,alpha=0.2,c=reward,cmap='viridis')
                plt.ylim(0,200)
                cbar = plt.colorbar()
                cbar.set_label('reward')
                plt.xlabel('x')
                plt.ylabel('y')
                #plt.title(title.split('.')[0] +' reward')
                plt.savefig(os.path.join(save_path,'reward',title.split('.')[0]+'.png'))
                plt.close()
                plt.rc('font', size=18) 
                plt.figure(figsize=(25,7))
                plt.plot(reward[:-1], label = 'reward')
                plt.plot(distance_prev_curr[:-1], label = 'distance prev curr')
                plt.plot(time_rew[:-1], label = 'time reward')
                plt.plot(distance_total_curr[:-1], label = 'distance total')
                plt.xlabel('time')
                plt.ylabel('reward')
                #plt.title(title.split('.')[0] +' reward')
                plt.legend()
                plt.savefig(os.path.join(save_path,'reward_split',title.split('.')[0]+'.png'))
                plt.close()
                plt.rc('font', size=18) 
                plt.figure(figsize=(16,9))
                plt.plot(distance_vehicle)
                plt.xlabel('time')
                plt.ylabel('distance [m]')
                plt.savefig(os.path.join(save_path,'distance_front_vehicle',title.split('.')[0]+'.png'))
                plt.close()
        if total == True:
            fields = ['episode', 'reward', 'total timesteps', 'max velocity', 'mean velocity', 'mean distance vehicle']
            with open(f'{save_path}/reward.csv','a') as reward_file:
                csvwriter = csv.writer(reward_file)
                csvwriter.writerow(fields)
                for i,re,t,v,v_m,d_v in zip(episode_list,reward_total, timestep_list,vel_list,d_p_c, d_v): 
                    if re < 700:
                        continue
                    row = [[i,re,t,v,v_m,d_v]]
                    csvwriter.writerows(row)
            plt.figure(figsize=(25,7))
            fig,ax1 = plt.subplots()
            
            color='tab:green'
            ax1.set_xlabel('episode')
            ax1.set_ylabel('reward')
            ax1.plot(reward_total, color=color)
            ax2 = ax1.twinx()
            color='tab:blue'
            ax2.set_ylabel('alpha')
            ax2.plot(alpha_list,color=color)
            fig.tight_layout()
            fig.savefig(os.path.join(save_path, 'reward_and_variance.png'))
            plt.close()
            plt.rc('font', size=18) 
            plt.figure(figsize=(25,7))
            plt.plot(alpha_list)
            plt.xlabel('episode')
            plt.ylabel('variance')
            plt.savefig(os.path.join(save_path, 'variance.png'))
            plt.close()
            plt.rc('font', size=18) 
            plt.figure(figsize=(25,5))
            plt.plot(velocity_dist)
            plt.xlabel('episode')
            plt.ylabel('velocity')
            #plt.title('velocity')
            plt.savefig(os.path.join(save_path,'velocity.png'))
            plt.close()
            #plt.rc('font', size=18) 
            # # plt.figure(figsize=(25,5))
            # # plt.plot(var_reward)
            
            # # plt.xlabel('episode')
            # # plt.ylabel('var reward')
            # # #plt.title('var reward'+f'{np.mean(var_reward)}')
            # # plt.savefig(os.path.join(save_path,'var_reward.png'))
            # # plt.close()
            plt.rc('font', size=18) 
            plt.figure(figsize=(25,7))
            plt.plot(brake_dist, label='brake')
            plt.plot(throttle_dist, label='throttle')
            plt.xlabel('episode')
            plt.ylabel('Distribution throttle/brake')
            plt.legend()
            #plt.title('Distribution throttle brake per episode')
            plt.savefig(os.path.join(save_path, 'distribution_throttle_brake', 'dist_throttle_brake_normalized.png'))
            plt.close()
            plt.rc('font', size=18) 
            plt.figure(figsize=(25,7))
            plt.plot(reward_total)
            #plt.title('reward over episode')
            plt.xlabel('episode')
            plt.ylabel('reward')
            plt.savefig(os.path.join(save_path, 'reward.png'))
            plt.close()   
            plt.rc('font', size=18) 
            plt.figure(figsize=(25,5))
            plt.plot(mean_reward, label='average reward')
            plt.plot(std_reward, label='std reward')
            plt.legend()
            #plt.title('average reward over episode')
            plt.xlabel('episode')
            plt.ylabel('average reward')
            plt.savefig(os.path.join(save_path, 'average_std_reward.png'))
            plt.close() 
            plt.rc('font', size=18) 
            plt.figure(figsize=(25,7))
            plt.plot(brake_or_throttle)
            #plt.title('brake or throttle over episode')
            plt.xlabel('episode')
            plt.ylabel('brake/throttle')
            plt.savefig(os.path.join(save_path, 'brake_or_throttle.png'))
            plt.close()
            #     #plot_scatter(x,y,save_path,title,x_label, y_label, z=vel)