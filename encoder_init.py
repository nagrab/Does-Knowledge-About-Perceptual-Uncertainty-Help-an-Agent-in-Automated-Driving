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
    #def __init__(self):
        # self.i = 0
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # try:
        #     self.decoder = Decoder(latent_dim).to(self.device)
        #     self.conv_encoder = VariationalEncoder(self.latent_dim).to(self.device)
        #     self.conv_encoder.eval()
        #     self.conv_encoder.load()
        #     self.decoder.load()
        #     self.decoder.eval()
            
        #     for params in self.conv_encoder.parameters():
        #         params.requires_grad = False
        # except:
        #     print('Encoder could not be initialized.')
        #     sys.exit()
    
    def process(self, observation, timestep):
        
        debugdelmedell = None
        #observation[0]= cv2.cvtColor(observation[0], cv2.COLOR_BGR2RGB)
        #cv2.imwrite(os.path.join('vae_train_during_rl_images',f'observation_img_{episode}.png'), observation[0])
        
        image_obs = torch.tensor(observation[0], dtype=torch.float).to(self.device) #Welche Farbe hat hier oberservation?
        #image_obs = image_obs.unsqueeze(0)
        #image_obs = image_obs.permute(0,3,2,1) # TODO: Double check if right dim for network
        #image_obs = image_obs.permute(0,3,1,2) # TODO: Double check if right dim for network
        #image_obs = self.conv_encoder(image_obs/255)
        #img_test = self.decoder(image_obs)
        # print('img_test')
        #img_test = img_test.cpu()
        #transform = T.ToPILImage()
        # print('transform')
        # print(img_test.size())
        #img_test = transform(img_test[0])
        # print('transformed')
        #img_test.save(os.path.join('vae_train_during_rl_images',f'vae_{episode}.png'))
        # self.i += 1 
        # if self.i == 100:
        #     self.i = 0
        # print('save')
        # if timestep % 1000 == 0:
        #     #plt.show(observation[0])
        #     Image.fromarray(observation[0]).show()
        #     img_test.show()
        navigation_obs = torch.tensor(observation[1], dtype=torch.float).to(self.device)
        #observation = torch.cat((image_obs.view(-1), navigation_obs), -1)
        observation = torch.cat((image_obs,navigation_obs),-1)
        return observation