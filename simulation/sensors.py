import math
import numpy as np
import weakref
import pygame
from simulation.connection import carla
from simulation.settings import RGB_CAMERA, SSC_CAMERA, ISSC_CAMERA
from PIL import Image
import cv2
import copy
import time
import simulation.pertubation
'''
Source:https://github.com/idreesshaikh/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning
'''
# ---------------------------------------------------------------------|
# ------------------------------- CAMERA |
# ---------------------------------------------------------------------|

class CameraSensor():

    def __init__(self, vehicle):
        #self.std = std                             #Std for Gaussian
        self.color = (142,0, 0)
        self.ego_position = (40,104)
        self.sensor_name = SSC_CAMERA
        self.sensor_name_instace = ISSC_CAMERA
        self.sensor_name_rgb = RGB_CAMERA
        self.parent = vehicle
        self.front_camera = list()
        self.front_camera_instance = list()
        self.front_rgb = list()
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        #self.sensor_rgb = self._set_camera_sensor(world,self.sensor_name_rgb)
        #self.sensor_instance = self._set_camera_sensor(world, self.sensor_name_instance)
        weak_self = weakref.ref(self)
        # self.sensor_instance.listen(
        #     lambda instance_image: CameraSensor._get_instance_camera_data(weak_self, instance_image)
        # )
        self.sensor.listen(
            lambda image: CameraSensor._get_front_camera_data(weak_self, image))
        # self.sensor_rgb.listen(
        #     lambda image: CameraSensor._get_rgb_camera_data(weak_self, image)
        # )
        self.tlc         = 0
        self.ids         = []
        self.ids_old     = []
        self.number_disp = []
        self.disap = True
        self.color_list = [142, 0, 0] #[b,g,r]
        self.color_id = 14 
        self.color_road = [128, 64,128]
        self.color_road_id = 1
        self.shifting = False
        self.cloning = False
        self.reshape_rotate = False
        self.time_line_disappearing = 5
        self.disappearing = False
    # Main front camera is setup and provide the visual observations for our network.
    def _set_camera_sensor(self, world, instance=None):
        if instance == None:
            front_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        else:
            front_camera_bp = world.get_blueprint_library().find(self.sensor_name_rgb)
        # front_camera_bp.set_attribute('image_size_x', f'80')
        # front_camera_bp.set_attribute('image_size_y', f'160')
        front_camera_bp.set_attribute('image_size_x', f'25')
        front_camera_bp.set_attribute('image_size_y', f'25')
        front_camera_bp.set_attribute('fov', f'90') #origin: 125 #me: 90
        front_camera = world.spawn_actor(front_camera_bp, carla.Transform(
            carla.Location(x=15, z=25), carla.Rotation(pitch= -90)), attach_to=self.parent) #origin: x=2.4,z= 1.5 pitch=-10 #me: x=0, z=15 pitch=-90 /x=51.61 for ego not in image x=15 for goldener schnitt
        return front_camera
    
    @staticmethod
    def _get_instance_camera_data(weak_self, image):
        self = weak_self()
        if not self:
            return
        placeholder = np.frombuffer(image.raw_data, dtype = np.dtype("uint8"))
        placeholder1 = placeholder.reshape((image.height, image.width, 4))
        self.front_camera_instance.append(placeholder1[:,:,:3])

    @staticmethod
    def _get_rgb_camera_data(weak_self,image):
        self = weak_self()
        if not self:
            return
        placeholder = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = placeholder.reshape((image.height, image.width, 4))
        self.front_rgb.append(placeholder1[:,:,:3])

    @staticmethod
    def _get_front_camera_data(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.CityScapesPalette)
        #image.save_to_disk("delme.png")
        placeholder = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        #print(np.shape(placeholder))
        # BUG? switched dims??
        #placeholder1 = placeholder.reshape((image.width, image.height, 4))
        placeholder1 = placeholder.reshape((image.height, image.width, 4))
        target = placeholder1[:, :, :3] #TODO: insert shfiting? 
        #Shfiting
        #if self.shifting == True:
        #   target = pertubation.shifting_objects(target, self.color, self.ego_position, target)
        #Cloning
        #if self.cloning == True:
        #   target = pertubation.cloning_objects(target, self.color, self.color_road)
        #Reshape/Rotation
        #if self.reshape_rotate == True:
        #t  arget = pertubation.reshape_objects(target, self.color, self.ego_position, target, self.color_id)
        # Disappearing
        #if self.disappearing == True:
            # if tlc % 2*self.time_line_dis ==0:
            #             tlc = 0
            # if tlc>self.time_line_dis and tlc<2*self.time_line_dis :
            #     disap = False
            #     tlc +=1
            #     number_disp = []
            #     ids = []
            #     ids_old = []
            # else:
            #     disap = True
            # if disap == True:
            #     ids, target = pertubation.disappear_objects(copy.deepcopy(target), self.color_list, self.color_id, self.ego_position, ids, instance_image)
            #     #Update ids and number of disappearing objects and counting the ids
            #     if len(ids_old)==0:
            #         ids_old = ids
            #         for i in range(len(ids_old)):
            #             number_disp.append(1)
            #     else:
            #         for k, id in enumerate(ids):
            #             if id in ids_old:
            #                 idx = ids_old.index(id)
            #                 number_disp[idx]+=1
            #                 if number_disp[idx]>self.time_line_dis:
            #                     number_disp.pop(idx)
            #                     ids.remove(id)
            #                     ids_old.remove(id)
            #             else:
            #                 ids_old.append(id)
            #                 number_disp.append(1)
            #         for k, id in enumerate(ids_old):
                        
            #             if id not in ids:
            #                 idx = ids_old.index(id)
            #                 number_disp.pop(idx)
            #                 ids_old.pop(idx)
            #     ids = ids_old
            #     tlc +=1
        #cv2.imwrite('target.png', target)
        #target = CameraSensor.shifting_objects(self,target) #TODO: Disappearing objects, clone objects, resize objects
        # print(target.shape)
        # img = Image.fromarray(target)
        # img.save("delmetarget.png")
        self.front_camera.append(target)#/255.0)

    # @staticmethod
    # def reshape_objects(self,oracle_image,  color, ego_position, gt_image):
    #     '''
    #     Reshape objects, like increasing or decreasing the object, or rotation it 90 degree

    #     Inputparameter:
    #     -------------------------------------------
    #     Oracle_image: Array: Array of the given image
    #     Color       : Array: Array of rgb values to find the object color (we are in the setting of semantic segmentation)
    #     Ego position: Array: Tuple of x and y for center of ego vehicle
    #     Gt_img      : Array: Array of ground truth

    #     Outputparameter:
    #     ---------------------------------------------
    #     Shifted_img : Array: Array or image with reshaped objects
    #     '''
    #     contours,_ = self.find_contours(oracle_image, color)
    #     contours_copy = copy.deepcopy(contours)
    #     for i,contour in enumerate(contours):
    #         if np.min(contour[:,0,0])<ego_position[0] and np.max(contour[:,0,0])>ego_position[0] and np.min(contour[:,0,1])<ego_position[1] and np.max(contour[:,0,1])> ego_position[1]:
    #             j = i
    #             continue
    #         else:
    #             reshape_x = bool(np.random.randint(2))
    #             reshape_y = bool(np.random.randint(2))
    #             rotate    = bool(np.random.randint(2))
    #         min_x = np.min(contour[:,0,0])
    #         min_y = np.min(contour[:,0,1])
    #         max_x = np.max(contour[:,0,0])
    #         max_y = np.max(contour[:,0,1])
    #         if reshape_x == True or reshape_y == True:
    #             if reshape_x == True and reshape_y == True:
    #                 shift_x = np.random.uniform(0.5,2)
    #                 shift_y = np.random.uniform(0.5,2)
    #             elif reshape_x == True:
    #                 shift_x = np.random.uniform(0.5,2)
    #                 shift_y = 1
    #             elif reshape_y == True:
    #                 shift_x = 1
    #                 shift_y = np.random.uniform(0.5,2)
    #             contour[:,0,0] = contour[:,0,0]*shift_x
    #             contour[:,0,1] = contour[:,0,1]*shift_y
    #             contours_copy = copy.deepcopy(contours)
    #         if rotate == True:
    #             contour[:,0,0] = contours_copy[i][:,0,1]
    #             contour[:,0,1] = contours_copy[i][:,0,0]
    #     list_contour = list(contours)
    #     try:
    #         list_contour.pop(j)
    #     except:
    #         pass
    #     contours = tuple(list_contour)
    #     filled_image, _ = self.fill_oracle_contours(contours, oracle_image, gt_image)
    #     shifted_image = cv2.fillPoly(filled_image, pts=contours, color=color)
        
    #     return shifted_image
    
    # @staticmethod
    # def fill_oracle_contours(self,contours, oracle_img, gt_img):
    #     '''
    #     Fill the oracle contours with a suitable color and delete the objects.

    #     Inputparameter:
    #     ---------------------------------------
    #     Contours  : Array: Array of lists return the edges of the contour polygons
    #     Oracle_img: Array: Input image where we want to find the objects

    #     Outputparameter:
    #     --------------------------------------
    #     Oracle_img: Array: Output with deleted objects
    #     '''
    #     for contour in contours:
    #         min_x = np.min(contour[:,0,0])
    #         max_x = np.max(contour[:,0,0])
    #         min_y = np.min(contour[:,0,1])
    #         max_y = np.max(contour[:,0,1])
    #         if min_x < 0:
    #             min_x = 0
    #         if max_x > np.shape(oracle_img)[1]-1:
    #             max_x = np.shape(oracle_img)[1]-1
    #         if min_y < 0:
    #             min_y = 0
    #         if max_y > np.shape(oracle_img)[0]-1:
    #             max_y = np.shape(oracle_img)[0]-1
    #         dist_x = abs(max_x-min_x)
    #         dist_y = abs(max_y-min_y)
    #         if dist_x > dist_y:
    #             if max_x == np.shape(oracle_img)[1]-1:
    #                     max_x -=1
    #             while min_y <= max_y:
                    
    #                 for x in range(min_x, max_x+1):
    #                     oracle_img[max_y,x] = oracle_img[max_y,x-1]
    #                     oracle_img[min_y,x] = oracle_img[min_y,x-1]
                    
    #                 min_y += 1
    #                 max_y -= 1                
    #         elif dist_x <= dist_y:
    #             if max_y == np.shape(oracle_img)[0]-1:
    #                 max_y -=1
    #             while min_x <=max_x:
    #                 for y in range(min_y, max_y+1):
    #                     oracle_img[y,max_x] = oracle_img[y-1, max_x]
    #                     oracle_img[y,min_x] = oracle_img[y-1, min_x]
                       
    #                 min_x += 1
    #                 max_x -= 1
                            
    #     return oracle_img

    # @staticmethod
    # def compute_shift(self,contours_shift,  ego_position):
    #     '''
    #     Compute Shift via multivariat Gaussian distribution for each object. 
    #     The script will compute first a mean for x and y shifting, and then it compute the true shifting, via analysing the shape of image.

    #     Inputparameter:
    #     ----------------------------------------------------------
    #     Contours_shift: Array: Array of lists return the edges of the contour polygons
    #     Ego position  : Array: Tuple of x and y for center of ego vehicle

    #     Outputparameter:
    #     ----------------------------------------------------------
    #     Contours_shift: Array: Array of lists return the updated/shifted edges of the contour polygons
    #     '''
    #     for contour in contours_shift:
    #         if np.min(contour[:,0,0])<ego_position[0] and np.max(contour[:,0,0])>ego_position[0] and np.min(contour[:,0,1])<ego_position[1] and np.max(contour[:,0,1])> ego_position[1]:
    #             continue
    #         shift = np.random.normal(scale = self.std, size = 2)
    #         x_shift_total = shift[1]*(abs(np.max(contour[:,0,0])-np.min(contour[:,0,0])))
    #         y_shift_total = shift[0]*(abs(np.max(contour[:,0,1])-np.min(contour[:,0,1])))
    #         contour[:,0,0]= contour[:,0,0]+x_shift_total
    #         contour[:,0,1]= contour[:,0,1]+y_shift_total

    #     return contours_shift


    # @staticmethod
    # def find_contours(self,oracle_img, color):
    #     '''
    #     Find contours for the objects which we want to shift.

    #     Inputparameter:
    #     ----------------------------------------------------
    #     Oracle_img: Array: Input image where we want to find the contours
    #     Color     : Array: Array of rgb values to find the object color (we are in the setting of semantic segmenation)

    #     Outputparameter:
    #     ----------------------------------------------------
    #     Contours  : Array: Array of lists return the edges of the contour polygons
    #     '''
    #     hsv_img = cv2.cvtColor(oracle_img, cv2.COLOR_BGR2HSV)
    #     white_img = np.array(np.ones((np.shape(oracle_img)[0], np.shape(oracle_img)[1],3))*255,np.uint8)
    #     color_hsv = cv2.cvtColor(np.array(np.reshape(color, (1,1,np.shape(color)[0])),np.uint8), cv2.COLOR_BGR2HSV)
    #     lower_bound = color_hsv[0][0]   
    #     upper_bound = lower_bound  
    #     masked = cv2.inRange(hsv_img, lower_bound, upper_bound)
    #     masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    #     masked_img = cv2.bitwise_and(white_img, masked )
    #     contours, _ = cv2.findContours(cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #     return contours


    # @staticmethod
    # def shifting_objects(self,source_path): #, color, ego_position):
    #     '''
    #     Main script for shifting object in a image.

    #     Inputparameter:
    #     -------------------------------------------
    #     Source_path : Str  : Path from the input image where the objects should be shifting
    #     Color       : Array: Array of rgb values to find the object color (we are in the setting of semantic segmentation)
    #     Ego position: Array: Tuple of x and y for center of ego vehicle
    #     '''
    #     oracle_image = source_path#cv2.imread(source_path)
    #     contours = self.find_contours(self,oracle_image, self.color)
    #     contours_shift = copy.deepcopy(contours)
    #     shift = self.compute_shift(self,contours_shift, self.ego_position)
    #     image_without_objects = self.fill_oracle_contours(self,contours, copy.deepcopy(oracle_image))
    #     shifted_image = cv2.fillPoly(image_without_objects, pts=shift, color=self.color)
        
    #     return shifted_image


# ---------------------------------------------------------------------|
# ------------------------------- ENV CAMERA |
# ---------------------------------------------------------------------|

class CameraSensorEnv:

    def __init__(self, vehicle):

        pygame.init()
        self.display = pygame.display.set_mode((720, 720),pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.sensor_name = RGB_CAMERA
        self.parent = vehicle
        self.surface = None
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensorEnv._get_third_person_camera(weak_self, image))

    # Third camera is setup and provide the visual observations for our environment.

    def _set_camera_sensor(self, world):

        thrid_person_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        thrid_person_camera_bp.set_attribute('image_size_x', f'720')
        thrid_person_camera_bp.set_attribute('image_size_y', f'720')
        third_camera = world.spawn_actor(thrid_person_camera_bp, carla.Transform(
            carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=-12.0)), attach_to=self.parent)
        return third_camera

    @staticmethod
    def _get_third_person_camera(weak_self, image):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = array.reshape((image.width, image.height, 4))
        placeholder2 = placeholder1[:, :, :3]
        placeholder2 = placeholder2[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(placeholder2.swapaxes(0, 1))
        self.display.blit(self.surface, (0, 0))
        pygame.display.flip()


class CameraRGBSensor:
    def __init__(self, vehicle):
        #self.std = std                             #Std for Gaussian
        self.color = (142,0, 0)
        self.ego_position = (40,104)
        self.sensor_name = SSC_CAMERA
        self.sensor_name_instace = ISSC_CAMERA
        self.sensor_name_rgb = RGB_CAMERA
        self.parent = vehicle
        self.front_camera = list()
        self.front_camera_instance = list()
        self.front_rgb = list()
        world = self.parent.get_world()
        #self.sensor = self._set_camera_sensor(world)
        self.sensor_rgb = self._set_camera_sensor(world)
        #self.sensor_instance = self._set_camera_sensor(world, self.sensor_name_instance)
        weak_self = weakref.ref(self)
        # self.sensor_instance.listen(
        #     lambda instance_image: CameraSensor._get_instance_camera_data(weak_self, instance_image)
        # )
        # self.sensor.listen(
        #     lambda image: CameraSensor._get_front_camera_data(weak_self, image))
        self.sensor_rgb.listen(
            lambda image: CameraRGBSensor._get_rgb_camera_data(weak_self, image)
        )
        self.tlc         = 0
        self.ids         = []
        self.ids_old     = []
        self.number_disp = []
        self.disap = True
        self.color_list = [142, 0, 0] #[b,g,r]
        self.color_id = 14 
        self.color_road = [128, 64,128]
        self.color_road_id = 1
        self.shifting = False
        self.cloning = False
        self.reshape_rotate = False
        self.time_line_disappearing = 5
        self.disappearing = False
    # Main front camera is setup and provide the visual observations for our network.
    def _set_camera_sensor(self, world, instance=None):
        front_camera_bp = world.get_blueprint_library().find(self.sensor_name_rgb)
        # front_camera_bp.set_attribute('image_size_x', f'80')
        # front_camera_bp.set_attribute('image_size_y', f'160')
        front_camera_bp.set_attribute('image_size_x', f'80')
        front_camera_bp.set_attribute('image_size_y', f'80')
        front_camera_bp.set_attribute('fov', f'90') #origin: 125 #me: 90
        front_camera = world.spawn_actor(front_camera_bp, carla.Transform(
            carla.Location(x=15, z=25), carla.Rotation(pitch= -90)), attach_to=self.parent) #origin: x=2.4,z= 1.5 pitch=-10 #me: x=0, z=15 pitch=-90 /x=51.61 for ego not in image x=15 for goldener schnitt
        return front_camera
    
    @staticmethod
    def _get_instance_camera_data(weak_self, image):
        self = weak_self()
        if not self:
            return
        placeholder = np.frombuffer(image.raw_data, dtype = np.dtype("uint8"))
        placeholder1 = placeholder.reshape((image.height, image.width, 4))
        self.front_camera_instance.append(placeholder1[:,:,:3])

    @staticmethod
    def _get_rgb_camera_data(weak_self,image):
        self = weak_self()
        if not self:
            return
        placeholder = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = placeholder.reshape((image.height, image.width, 4))
        self.front_rgb.append(placeholder1[:,:,:3])


class CamerafrontSensor:
    def __init__(self, vehicle):
        #self.std = std                             #Std for Gaussian
        self.color = (142,0, 0)
        self.ego_position = (40,104)
        self.sensor_name = SSC_CAMERA
        self.sensor_name_instace = ISSC_CAMERA
        self.sensor_name_rgb = RGB_CAMERA
        self.parent = vehicle
        self.front_camera = list()
        self.front_camera_instance = list()
        self.front_front = list()
        world = self.parent.get_world()
        #self.sensor = self._set_camera_sensor(world)
        self.sensor_front = self._set_camera_sensor(world)
        #self.sensor_instance = self._set_camera_sensor(world, self.sensor_name_instance)
        weak_self = weakref.ref(self)
        # self.sensor_instance.listen(
        #     lambda instance_image: CameraSensor._get_instance_camera_data(weak_self, instance_image)
        # )
        # self.sensor.listen(
        #     lambda image: CameraSensor._get_front_camera_data(weak_self, image))
        self.sensor_front.listen(
            lambda image: CamerafrontSensor._get_front_camera_data(weak_self, image)
        )
        self.tlc         = 0
        self.ids         = []
        self.ids_old     = []
        self.number_disp = []
        self.disap = True
        self.color_list = [142, 0, 0] #[b,g,r]
        self.color_id = 14 
        self.color_road = [128, 64,128]
        self.color_road_id = 1
        self.shifting = False
        self.cloning = False
        self.reshape_rotate = False
        self.time_line_disappearing = 5
        self.disappearing = False
    # Main front camera is setup and provide the visual observations for our network.
    def _set_camera_sensor(self, world, instance=None):
        front_camera_bp = world.get_blueprint_library().find(self.sensor_name_rgb)
        # front_camera_bp.set_attribute('image_size_x', f'80')
        # front_camera_bp.set_attribute('image_size_y', f'160')
        front_camera_bp.set_attribute('image_size_x', f'80')
        front_camera_bp.set_attribute('image_size_y', f'80')
        #front_camera_bp.set_attribute('fov', f'90') #origin: 125 #me: 90
        front_camera = world.spawn_actor(front_camera_bp, carla.Transform(
            carla.Location(x=1.6, z=1.7)), attach_to=self.parent) #origin: x=2.4,z= 1.5 pitch=-10 #me: x=0, z=15 pitch=-90 /x=51.61 for ego not in image x=15 for goldener schnitt
        return front_camera
    
    @staticmethod
    def _get_instance_camera_data(weak_self, image):
        self = weak_self()
        if not self:
            return
        placeholder = np.frombuffer(image.raw_data, dtype = np.dtype("uint8"))
        placeholder1 = placeholder.reshape((image.height, image.width, 4))
        self.front_camera_instance.append(placeholder1[:,:,:3])

    @staticmethod
    def _get_front_camera_data(weak_self,image):
        self = weak_self()
        if not self:
            return
        placeholder = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = placeholder.reshape((image.height, image.width, 4))
        self.front_front.append(placeholder1[:,:,:3])
# ---------------------------------------------------------------------|
# ------------------------------- COLLISION SENSOR|
# ---------------------------------------------------------------------|

# It's an important as it helps us to tract collisions
# It also helps with resetting the vehicle after detecting any collisions
class CollisionSensor:

    def __init__(self, vehicle) -> None:
        self.sensor_name = 'sensor.other.collision'
        self.parent = vehicle
        self.collision_data = list()
        world = self.parent.get_world()
        self.sensor = self._set_collision_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    # Collision sensor to detect collisions occured in the driving process.
    def _set_collision_sensor(self, world) -> object:
        collision_sensor_bp = world.get_blueprint_library().find(self.sensor_name)
        sensor_relative_transform = carla.Transform(
            carla.Location(x=1.3, z=0.5))
        collision_sensor = world.spawn_actor(
            collision_sensor_bp, sensor_relative_transform, attach_to=self.parent)
        return collision_sensor

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_data.append(intensity)

