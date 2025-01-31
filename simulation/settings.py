'''
All the parameters used in the Simulation has been documented here.

Easily modifiable paramters with the quick access in this settings.py file \
    to achieve quick modifications especially during the training sessions.

Names of the parameters are self-explanatory therefore elimating the use of further comments.
'''
'''
Source:https://github.com/idreesshaikh/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning
'''

HOST = "localhost"
PORT = 2000
TIMEOUT = 20.0

CAR_NAME = 'model3'
EPISODE_LENGTH = 750
NUMBER_OF_VEHICLES = 35   #This number can be modify but depends on the technical resources that are used. In case of more resource the number can be increased.
NUMBER_OF_PEDESTRIAN = 20
CONTINUOUS_ACTION = True
VISUAL_DISPLAY = False


RGB_CAMERA = 'sensor.camera.rgb'
SSC_CAMERA = 'sensor.camera.semantic_segmentation'
ISSC_CAMERA = 'sensor.camera.instance_segmentation'

