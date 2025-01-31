import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
def read_npy(path):
    '''
    Read npy file
    Inputparameter:
    ------------------------------------------
    path: str: Path of the npy file

    Outputparameter:
    ------------------------------------------
    npy_file: np array: Array with data
    '''
    return np.load(path, allow_pickle=True)
def compute_statistical_values(x_m,y_m):
    mean_x = np.mean(x_m)
    mean_y = np.mean(y_m)


    return (mean_x, mean_y)
def plot_scatter(npy_file, title, save_path, xlabel, ylabel):
    '''
    Plot scatter plot 
    npy_file : np array: Array with data
    title    :      str: Title for the plot
    save_path:      str: Save path for the plot
    xlabel   :      str: Name of x axis
    ylabel   :      str: Name of y axis
    '''
    x = npy_file[1][:]
    y = npy_file[0][:]
    plt.figure(figsize=(16,9))
    plt.scatter(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    title_second = compute_statistical_values(x,y)
    #plt.title(title+f' mean {xlabel}: '+ str(round(title_second[0],3))+f' mean {ylabel}: '+str(round(title_second[1],3)))
    plt.savefig(save_path+'/'+'reward_value/'+title+'.png')
    plt.close()

    plt.scatter(x,y-x)
    plt.xlabel(xlabel)
    plt.ylabel('advantage')
    title_second = compute_statistical_values(x,y-x)
    #plt.title(title+' advantage'+ f' mean {xlabel}: '+str(round(title_second[0],3))+f' mean advantage: '+str(round(title_second[1],3)))
    plt.savefig(save_path+'/'+'advantage_value/'+title+'_advantage_'+xlabel+'.png')
    plt.close()

def plot_advantage_over_time(npy_file,save_path, ylabel, title, close = False, plotting = None,normalized = False):
    plt.rc('font', size=18) 
    plt.figure(figsize=(16,9))
    if normalized == True:
        plotting = (plotting-np.mean(plotting))/(np.std(plotting)+1e-7)
    plt.plot(plotting)#[::-1])
    plt.xlabel('time')
    plt.ylabel(ylabel)
    #plt.title(title+' '+ ylabel+' '+f'{np.mean(plotting)}')
    plt.savefig(save_path+title+f'_{ylabel}'+'.png')
    if close == True:
        plt.close()


def main(path, save_path, xlabel, ylabel):
    npy_file = read_npy(path)
    y = npy_file[0][:]
    # x = npy_file[1][:]
    # z = npy_file[2][:]
    # v = npy_file[3][:]
    ratio = npy_file[4][:]
    #y_new = x-z
    #plot_advantage_over_time(npy_file,save_path, ylabel, path.split('/')[-1].split('.')[0],close=True, plotting =y_new)
    #plot_scatter(npy_file, path.split('/')[-1].split('.')[0],save_path, xlabel, ylabel )
    return y[0], npy_file[4][:][-1], npy_file[3][:] #npy_file[1][:], npy_file[0][:], npy_file[2][:], 
    
if __name__ == '__main__':
    path = '/home/carla-admin/ma-grabowsky/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning/reward_advantageTown02_train_wo_pert_and_unc_lprev_lcurr_times_1_t_tmax_min_d_mal_50_new_br_th_train_each_eps_tim_mal_ga_0.999_va_0.1_500000.0_0.025_0.025_nw_img_nw_inertia_3_2_7500_1e-05'
    save_path = '/home/carla-admin/ma-grabowsky/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning/vae_with_shifting/Town02_train_wo_pert_and_unc_lprev_lcurr_times_1_t_tmax_min_d_mal_50_new_br_th_train_each_eps_tim_mal_ga_0.999_va_0.1_500000.0_0.025_0.025_nw_img_nw_inertia_3_2_7500_1e-05'
    if os.path.exists(save_path) == False:
        os.makedirs(save_path+'/'+'reward_value/')
        os.makedirs(save_path+'/'+'advantage_value/')
    if os.path.exists(save_path+'advantage/')==False:
        os.makedirs(save_path+'advantage/')
    if os.path.exists(save_path+'value/')==False:
        os.makedirs(save_path+'value/')
    if os.path.exists(save_path+'reward_norm/')==False:
        os.makedirs(save_path+'reward_norm/')
    if os.path.exists(save_path+'dis_reward/')==False:
        os.makedirs(save_path+'dis_reward/')
    if os.path.exists(save_path+'loss/')==False:
        os.makedirs(save_path+ 'loss/')
    xlabel = 'value'
    ylabel = 'advantage'
    name = 'advantage'
    rew_dis = np.array([0])
    val_lis = np.array([0])
    rew_lis = np.array([0])
    loss_lis = np.array([0])
    ratio_lis_value = np.array([0])
    ratio_lis = []
    ratio_var_lis = []
    explained_variance = []
    ratio_var_list_plus=[]
    #ratio_var_lis = np.array([0])
    average_return = []
    total_return = []
    for root, dir, files in os.walk(path):
        print(files)
        if len(files) == 0:
            continue
        for i in tqdm(range(len(files))):
            if i == 1:
                continue
            ret,r,v = main(root+'/'+f'{i}'+'.npy', save_path+f'{name}/', xlabel, ylabel) #x,y,z,v,
            # rew_dis = np.concatenate((rew_dis, y))
            # val_lis = np.concatenate((val_lis,z))
            # rew_lis = np.concatenate((rew_lis,x))
            loss_lis = np.concatenate((loss_lis,v))
            #explained_variance.append((1-np.var(x-z))/np.var(x))
            ratio_lis.append(np.mean(r))
            ratio_var_lis.append(np.mean(r)-np.std(r))
            ratio_var_list_plus.append(np.mean(r)+np.std(r))
            ratio_lis_value = np.concatenate((ratio_lis_value,r))
            total_return.append(ret)
            average_return.append(np.mean(total_return))

    plt.rc('font', size=20)
    plt.figure(figsize=(25,2.5))
    plt.plot(average_return)
    plt.xlabel('episode')
    plt.ylabel('average return')
    plt.savefig(save_path+'average_return.png')
    plt.close()
            # ratio_var_lis = np.concatenate((ratio_var_lis,np.var(r)))
    # advantage = rew_lis-val_lis
    # advantage_mean = []
    # for i in range(len(advantage)):
    #     advantage_mean.append(np.mean(advantage[:i+1]))
    # plt.rc('font', size=18) 
    # plt.figure(figsize=(25,2.5))
    # plt.plot(advantage_mean)
    # plt.xlabel('time')
    # plt.ylabel(ylabel)
    # #plt.title('total'+f' {name}')#+f'{np.mean(rew_lis-val_lis)}')
    # plt.savefig(save_path+'mean'+f'_{name}_{advantage_mean[-1]}'+'.png')
    # plt.close()
    # plt.rc('font', size=18) 
    # plt.figure(figsize=(25,2.5))
    # ratio_new = []
    # ratio_new_value = []
    # for ratio in ratio_lis:
    #     if ratio < 0.8:
    #         ratio=0.8
    #     elif ratio > 1.2:
    #         ratio=1.2
    #     ratio_new.append(ratio)
    # for ratio in ratio_lis_value:
    #     if ratio<0.8:
    #         ratio=0.8
    #     elif ratio > 1.2:
    #         ratio=1.2
    #     ratio_new_value.append(ratio)
    # plt.plot(ratio_new_value)
    # plt.xlabel('time')
    # plt.ylabel('ratio')
    # plt.savefig(save_path+'total_ratio_value.png')
    # plt.close()
    # plt.plot(ratio_new)
    # plt.xlabel('time')
    # plt.ylabel(ylabel)
    # #plt.title('total'+f' {name}')#+f'{np.mean(rew_lis-val_lis)}')
    # plt.savefig(save_path+'total'+f'_ratio'+'.png')
    # plt.close()
    # plt.plot(ratio_lis)#
    # plt.plot(ratio_var_lis, alpha=0.2,color='orange')
    # plt.plot(ratio_var_list_plus, alpha=0.2, color='orange')
    # #plt.fill_between(ratio_lis,ratio_var_list_plus, ratio_var_lis, color='orange')
    # plt.axhline(y=1.2,color='r', linestyle='-')
    # plt.axhline(y=0.8,color='r', linestyle='-')
    # plt.xlabel('time')
    # plt.ylabel('ratio')
    # plt.savefig(save_path+'total'+'_ratio_complete.png')
    # plt.close()
    # mean_loss = []
    # for i in range(len(loss_lis)):
    #     mean_loss.append(np.mean(loss_lis[:i+1]))
    # plt.figure(figsize=(25,5))
    # plt.plot(loss_lis)
    # plt.plot(mean_loss)
    # plt.xlabel('time')
    # plt.ylabel('loss')
    # plt.title('total loss')
    # plt.savefig(save_path+'total_loss.png')
    # plt.close()
    # plt.plot(mean_loss)
    # plt.xlabel('time')
    # plt.ylabel('mean loss')
    # plt.savefig(save_path+f'mean_loss_{mean_loss[-1]}.png')
    # plt.close()
    # plt.plot(explained_variance)
    # plt.xlabel('episode')
    # plt.ylabel('variance')
    # plt.title('explained variance')
    # plt.savefig(save_path+'explained_variance.png') 
    # plt.close()   
