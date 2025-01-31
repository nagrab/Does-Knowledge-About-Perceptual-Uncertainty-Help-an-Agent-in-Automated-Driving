# Does_Uncertainty_Estimation_for_Perception_in_Automated_Driving_Help_for_Planning
This repo contains the scripts for training and validating the reinforcement learner with PPO. 
continuous_driver.py is the main script that contains the initialisation of CARLA, obtaining the observation and actions. 



## Start training:
```
python continuous_driver.py --train True --town Town02
```
with checkpoint:
```
python continuous_driver.py --train True --town Town02 --load-checkpoint True
```
with pretrained weights:
```
python continuous_driver.py --train True --town Town02 --load-checkpoint True --done episode_number --run_list True --runpath /path/to/checkpoint/
```

## Installation
Install Carla 15 Release: Explanation and steps can be found at: https://carla.readthedocs.io/en/0.9.15/start_quickstart/ \
For this project Unreal Engine 4.26 is used. 
Install the Git Repo
```
cd existing_repo
git remote add origin https://github.com/nagrab/Does-Knowledge-About-Perceptual-Uncertainty-Help-an-Agent-in-Automated-Driving.git
git branch -M main
git push -uf origin main
```
