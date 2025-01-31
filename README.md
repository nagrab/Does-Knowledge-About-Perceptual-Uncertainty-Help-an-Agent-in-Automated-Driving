# Does Uncertainty Estimation for Perception in Automated Driving Help for Planning
Agents in real-world scenarios like automated driv-
ing deal with uncertainty in their environment, in particular
due to perceptual uncertainty. Although, reinforcement learning
is dedicated to autonomous decision-making under uncertainty
these algorithms are typically not informed about the uncertainty
currently contained in their environment. On the other hand,
uncertainty estimation for perception itself is typically directly
evaluated in the perception domain, e.g., in terms of false
positive detection rates or calibration errors based on camera
images. Its use for deciding on goal-oriented actions remains
largely unstudied. In this paper, we investigate how an agent’s
behavior is influenced by an uncertain perception and how
this behavior changes if information about this uncertainty is
available. Therefore, we consider a proxy task, where the agent is
rewarded for driving a route as fast as possible without colliding
with other road users. For controlled experiments, we introduce
uncertainty in the state space by perturbing the perception of
the given agent while informing the latter. Our experiments show
that an unreliable observation space modeled by a perturbed
perception leads to a conservative driving behavior of the agent.
Furthermore, when adding the information about the current
uncertainty directly to the state space, the agent adapts to the
specific situation and in general accomplishes its task faster while,
at the same time, accounting for risks\\

## Information
This reposity gives all relevant codes for reproduce the training, validation and testing process of the experiments from "Does Knowledge About Perceptual Uncertainty Help an Agent in Automated Driving". 
The code based on the reposity of Razak: https://github.com/idreesshaikh/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning



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
For this project Unreal Engine 4.26 is used. \
Install the Git Repo
```
cd existing_repo
git remote add origin https://github.com/nagrab/Does-Knowledge-About-Perceptual-Uncertainty-Help-an-Agent-in-Automated-Driving.git
git branch -M main
git push -uf origin main
```
