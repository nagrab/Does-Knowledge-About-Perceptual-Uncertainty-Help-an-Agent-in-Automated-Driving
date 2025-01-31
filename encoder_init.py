import sys
import torch
import torchvision
from autoencoder.encoder import VariationalEncoder
from autoencoder.decoder import Decoder
import cv2 
import torchvision.transforms as T
from PIL import Image
import os
import matplotlib.pyplot as plt
'''
Source: https://github.com/idreesshaikh/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning
'''
class EncodeState():
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
    def process(self, observation, timestep):
        
        debugdelmedell = None        
        image_obs = torch.tensor(observation[0], dtype=torch.float).to(self.device) #Welche Farbe hat hier oberservation?
        navigation_obs = torch.tensor(observation[1], dtype=torch.float).to(self.device)
        observation = torch.cat((image_obs,navigation_obs),-1)
        return observation
