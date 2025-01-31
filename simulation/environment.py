import time
import random
import numpy as np
import pygame
import cv2
import copy
from simulation.connection import carla
from simulation.sensors import CameraSensor, CameraSensorEnv, CollisionSensor,CameraRGBSensor, CamerafrontSensor
from simulation.settings import *
import csv
import os
from parameters import *

'''
Source: https://github.com/idreesshaikh/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning
'''
class CarlaEnvironment():

    def __init__(self, client, world, town, checkpoint_frequency=100, continuous_action=True) -> None:

        self.location_list = []
        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.action_space = self.get_discrete_action_space()
        self.continous_action_space = continuous_action
        self.display_on = VISUAL_DISPLAY
        self.vehicle = True
        self.settings = True
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0
        self.fresh_start=True
        self.checkpoint_frequency = checkpoint_frequency
        self.route_waypoints = None
        self.town = town
        self.spawn_points=[]
        self.other_vehicles = []
        # Objects to be kept alive
        self.camera_obj = None
        self.rgb_camera = None
        self.front_camera = None
        self.env_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        # self.waypoint_reward_75radius = None
        # self.waypoint_reward_5radius = None
        # self.waypoint_reward_25radius = None
        # self.waypoint_reward_1radius = None
        # self.waypoint_reward_index_75radius = 0
        # self.waypoint_reward_index_5radius = 0
        # self.waypoint_reward_index_25radius = 0
        # self.waypoint_reward_index_1radius = 0
        # Two very important lists for keeping track of our actors and their observations.
        vegetation = world.get_environment_objects(carla.CityObjectLabel.Vegetation)
        world.enable_environment_objects([*map(lambda obj: obj.id, vegetation)], False)
        self.sensor_list = list()
        self.actor_list = list()
        self.walker_list = list()
        # self.create_pedestrians()
        # self.set_other_vehicles()
       
        self.reward_velocity = 0
        self.last_waypoint = None
        self.distance=[]
    # A reset function for reseting our environment.
    def reset(self,episode, val = False, case = 'e',case_name='e'):
        print(case)
        try:
            
            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()
                #self.set_other_vehicles()
                
            self.remove_sensors()


            # Blueprint of our main vehicle
            vehicle_bp = self.get_vehicle(CAR_NAME)

            if self.town == "Town07":
                transform = self.map.get_spawn_points()[38] #Town7  is 38 
                self.total_distance = 250
            elif self.town == "Town02":
                
               
                if not os.path.exists(f'spawn_points_{self.town}_{SAVE_NAME}') :
                    all_spawn_points = self.map.get_spawn_points()
                    print('spawn point path generation')
                    os.makedirs(f'spawn_points_{self.town}_{SAVE_NAME}')
                    spawn_path = f'spawn_points_{self.town}_{SAVE_NAME}'
                    fields = ['index', 'x', 'y']
                    with open(f'{spawn_path}/spawn_points_{self.town}_{SAVE_NAME}.csv', 'a') as spawn_file:
                        csvwriter = csv.writer(spawn_file)
                        csvwriter.writerow(fields)
                        for i,sp in enumerate(all_spawn_points):
                            row = [[i, sp.location.x, sp.location.y]]
                            csvwriter.writerows(row)
                    print('spawn point written down')
                transform = self.map.get_spawn_points()[29] #Town2 is 1
                #print(transform.location)
                self.total_distance = 150
            elif self.town == "Town04":
                transform = self.map.get_spawn_points()[25]
                self.total_distance = 100
            else:
                transform = self.map.get_spawn_points()[1]#random.choice(self.map.get_spawn_points())
                self.total_distance = 250

            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            self.actor_list.append(self.vehicle)
            print('spawn vehicles')
            self.set_other_vehicles()
            self.std = 1
            # Camera Sensor
            self.camera_obj = CameraSensor(self.vehicle)
            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0002)
            self.image_obs = self.camera_obj.front_camera.pop(-1)#BGR
            self.sensor_list.append(self.camera_obj.sensor)

            self.rgb_camera = CameraRGBSensor(self.vehicle)
            while(len(self.rgb_camera.front_rgb)==0):
                time.sleep(0.0002)
            self.rgb_obs = self.rgb_camera.front_rgb.pop(-1)
            self.sensor_list.append(self.rgb_camera.sensor_rgb)

            self.front_camera = CamerafrontSensor(self.vehicle)
            while(len(self.front_camera.front_front)==0):
                time.sleep(0.0002)
            self.front_obs = self.front_camera.front_front.pop(-1)
            self.sensor_list.append(self.front_camera.sensor_front)
            # Third person view of our vehicle in the Simulated env
            # if self.display_on:
            #     self.env_camera_obj = CameraSensorEnv(self.vehicle)
            #     self.sensor_list.append(self.env_camera_obj.sensor)

            # Collision sensor
            self.collision_obj = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)

            
            self.timesteps = 0
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.previous_location = self.vehicle.get_location()
            self.spawn_location = self.vehicle.get_location()
            self.distance_traveled = 0.0
            self.center_lane_deviation = 0.0
            self.target_speed = 22 #km/h
            self.max_speed = 25.0
            self.min_speed = 0.0
            self.max_distance_from_center = 3#2.4
            self.throttle = float(1.0)
            self.brake = float(0.0)
            self.previous_steer = float(0.0)
            self.velocity = float(0.0)
            self.velocity_front = float(0.0)
            self.distance_from_center = float(0.0)
            self.angle = float(0.0)
            self.center_lane_deviation = 0.0
            self.distance_covered = 0.0
            self.distance = []

            if self.fresh_start:
                self.current_waypoint_index = 0
                # Waypoint nearby angle and distance from it
                self.route_waypoints = list()
                self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
                current_waypoint = self.waypoint
                self.route_waypoints.append(current_waypoint)
                #if self.waypoint_reward_75radius == None: ### TODO: kann weg  überall wo tot -> self.fresh_start=True
                
                
                for x in range(self.total_distance):
                    if self.town == "Town07":
                        if x < 200:
                            next_waypoint = current_waypoint.next(1.0)[0]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                    elif self.town == "Town02" or self.town=="Town04":
                        if x < 150:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[0]
                    else:
                        next_waypoint = current_waypoint.next(1.0)[0]
                    
                    self.route_waypoints.append(next_waypoint)
                    current_waypoint = next_waypoint
                    row= [[next_waypoint.transform.location.x, next_waypoint.transform.location.y]]
                    fields = ['x','y']
                    with open(os.path.join(SAVE_PATH,f'route_{self.town}_{SAVE_NAME}'), 'a') as file:
                        csvwriter = csv.writer(file)
                        if not os.path.exists(os.path.join(SAVE_PATH,f'route_{self.town}_{SAVE_NAME}')) or os.stat(os.path.join(SAVE_PATH,f'route_{self.town}_{SAVE_NAME}')).st_size == 0:
                            csvwriter.writerow(fields)
                        csvwriter.writerows(row)
                self.last_waypoint = current_waypoint
                # self.waypoint_reward_75radius = self.route_waypoints[1%len(self.route_waypoints)]
                # self.waypoint_reward_5radius = self.route_waypoints[1%len(self.route_waypoints)]
                # self.waypoint_reward_25radius = self.route_waypoints[1%len(self.route_waypoints)]
                # self.waypoint_reward_1radius = self.route_waypoints[1%len(self.route_waypoints)]


            else:
                # Teleport vehicle to last checkpoint
                waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
                transform = waypoint.transform
                self.vehicle.set_transform(transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index
            try:
                distance_31 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[1].get_location())[0:2])
            except:
                distance_31 = 200
                print('No NPC on spawning point 31')
            try:
                distance_4 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[16].get_location())[0:2])
            except:
                distance_4 = 200
                print('No NPC on spawning point 4')
            try:
                distance_25 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[7].get_location())[0:2])
            except:
                distance_25 = 200
                print('No NPC on spawning point 25')
            try:
                distance_20 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[9].get_location())[0:2])
            except:
                distance_20 = 200
                print('No NPC on spawning point 20')
            try:
                distance_ego_front_vehicle = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[5].get_location())[0:2])
            except:
                distance_ego_front_vehicle = 200
                print('No NPC on spawning point 27')
            if case == 'a':
                if distance_ego_front_vehicle <=42:
                    self.distance.append(distance_ego_front_vehicle)
            if case == 'b' :
                if distance_31<15:
                    self.distance.append(-distance_31)
                if distance_4 <15:
                    self.distance.append(-distance_4)
            if case == 'c':
                if distance_25 <= 42:
                    self.distance.append(distance_25)
                if distance_20 <=42:
                    self.distance.append(distance_20)
                if distance_ego_front_vehicle <= 42:
                    self.distance.append(distance_ego_front_vehicle)
            
            if case == 'd':
                if distance_31<15:
                    self.distance.append(-distance_31)
                if distance_4 <15:
                    self.distance.append(-distance_4)
                if distance_25 <= 42:
                    self.distance.append(distance_25)
                if distance_20 <=42:
                    self.distance.append(distance_20)
                if distance_ego_front_vehicle <= 42:
                    self.distance.append(distance_ego_front_vehicle)
            # elif spawning_case == 3:
            #     continue
               
                   
            # for wp in self.route_waypoints:
            #     self.world.debug.draw_point(wp.transform.location, size=0.05,life_time=0, color = carla.Color(0,255,255,0))
            #

                        
            time.sleep(0.5)
            self.collision_history.clear()

            self.episode_start_time = time.time()
            self.image_obs = self.image_obs[0:25, 10:14]
            # if val == False:
            #     if not os.path.exists(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}', f'episode_{episode+1}')) or os.stat(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}',f'episode_{episode+1}')).st_size == 0:
            #         os.makedirs(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}',f'episode_{episode+1}'))
            #         # os.makedirs(f'observation_{self.town}_{SAVE_NAME}/episode_{episode+1}_rgb')
            #         # os.makedirs(f'observation_{self.town}_{SAVE_NAME}/episode_{episode+1}_front')
            #     cv2.imwrite(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}',f'episode_{episode+1}','obs_img_0.png'),self.image_obs)
            # else:
            #     if not os.path.exists(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}_validation_{case_name}_{TNAME}_{RUN_NUMBER}',f'episode_{episode+1}_e')) or os.stat(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}_validation_{case_name}_{TNAME}_{RUN_NUMBER}',f'episode_{episode+1}_e')).st_size == 0:
            #         os.makedirs(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}_validation_{case_name}_{TNAME}_{RUN_NUMBER}',f'episode_{episode+1}_e'))
            #         print(f'Path generated: observation_{self.town}_{SAVE_NAME}_validation_{case_name}_{TNAME}_{RUN_NUMBER}/episode_{episode+1}_e')
            #         # os.makedirs(f'observation_{self.town}_{SAVE_NAME}_validation/episode_{episode+1}_rgb')
            #         # os.makedirs(f'observation_{self.town}_{SAVE_NAME}_validation/episode_{episode+1}_front')
            #     cv2.imwrite(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}_validation_{case_name}_{TNAME}_{RUN_NUMBER}',f'episode_{episode+1}_e','obs_img_0.png'),self.image_obs)
            if case != 'e' and len(self.distance) > 0:
                self.image_obs = self.disappear_object(self.image_obs, self.distance)
            # if val == False:
            #     if not os.path.exists(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}',f'episode_{episode+1}')) or os.stat(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}',f'episode_{episode+1}')).st_size == 0:
            #         os.makedirs(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}',f'episode_{episode+1}'))
            #         # os.makedirs(f'observation_{self.town}_{SAVE_NAME}/episode_{episode+1}_rgb')
            #         # os.makedirs(f'observation_{self.town}_{SAVE_NAME}/episode_{episode+1}_front')
            #     cv2.imwrite(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}',f'episode_{episode+1}','obs_img_0.png'),self.image_obs)
            #     # cv2.imwrite(os.path.join(f'observation_{self.town}_{SAVE_NAME}',f'episode_{episode+1}_rgb','obs_img_0.png'),self.rgb_obs)
            #     # cv2.imwrite(os.path.join(f'observation_{self.town}_{SAVE_NAME}',f'episode_{episode+1}_front','obs_img_0.png'),self.front_obs)
            # else:
            #     if not os.path.exists(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}_validation_{case_name}_{TNAME}_{RUN_NUMBER}',f'episode_{episode+1}')) or os.stat(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}_validation_{case}_{TNAME}_{RUN_NUMBER}',f'episode_{episode+1}')).st_size == 0:
            #         os.makedirs(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}_validation_{case_name}_{TNAME}_{RUN_NUMBER}',f'episode_{episode+1}'))
            #         print(f'Path generated: observation_{self.town}_{SAVE_NAME}_validation_{case_name}_{TNAME}_{RUN_NUMBER}/episode_{episode+1}')
            #         # os.makedirs(f'observation_{self.town}_{SAVE_NAME}_validation/episode_{episode+1}_rgb')
            #         # os.makedirs(f'observation_{self.town}_{SAVE_NAME}_validation/episode_{episode+1}_front')
            #     cv2.imwrite(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}_validation_{case_name}_{TNAME}_{RUN_NUMBER}',f'episode_{episode+1}','obs_img_0.png'),self.image_obs)
                # cv2.imwrite(os.path.join(f'observation_{self.town}_{SAVE_NAME}_validation',f'episode_{episode+1}_rgb','obs_img_0.png'),self.rgb_obs)
                # cv2.imwrite(os.path.join(f'observation_{self.town}_{SAVE_NAME}_validation',f'episode_{episode+1}_front','obs_img_0.png'),self.front_obs)
                
            self.image_obs = cv2.cvtColor(self.image_obs, cv2.COLOR_RGB2GRAY)
            self.image_obs = self.image_obs.flatten()
            #self.image_obs= cv2.cvtColor(self.image_obs, cv2.COLOR_RGB2BGR)
            print(case)
            ########Uncertainty
            a = 0
            b = 0
            c = 0
            d = 0
            e = 0
            if case == 'a':
                a = 1
            elif case == 'b':
                b = 1
            elif case == 'c':
                c = 1
            elif case == 'd':
                d = 1
            elif case == 'e':
                e = 1

            self.navigation_obs = np.array([self.throttle, self.velocity,self.previous_steer, self.distance_from_center, self.angle, self.brake, a, b, c, d])
            print('Feedback reset')
            return [self.image_obs, self.navigation_obs]#, spawning_case
            #return self.navigation_obs
        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            #self.set_other_vehicles()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()


# ----------------------------------------------------------------
# Step method is used for implementing actions taken by our agent|
# ----------------------------------------------------------------

    # A step function is used for taking inputs generated by neural net.
    def step(self, action_idx,episode, val = False, case = 'e',case_name='e'):
        #xodr_map = open('xodr_map.xodr', 'a')
        # self.map.cook_in_memory_map('Town_02_xodr.xodr')
        try:
            self.distance = []
            finished_route = False
            self.timesteps+=1
            self.fresh_start = False
            # Velocity of the vehicle
            velocity = self.vehicle.get_velocity()
            self.velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            velocity_front = self.actor_list[5].get_velocity()
            self.velocity_front = np.sqrt(velocity_front.x**2+velocity_front.y**2+velocity_front.z**2)*3.6
            
            # Action fron action space for contolling the vehicle with a discrete action
            if self.continous_action_space:
                # steer = float(action_idx[0])
                # steer = max(min(steer, 1.0), -1.0)
                steer = 0.0
                #throttle = float((action_idx[1] + 1.0)/2)
                throttle = float((action_idx[0] + 1.0)/2)
                throttle = max(min(throttle, 1.0), 0.0)
                #brake = float((action_idx[2]+1.0)/2)
                brake = float((action_idx[1]+1.0)/2)
                brake = max(min(brake,1.0), 0.0)
                
                # velocity_modification = float(action_idx[0])
                # velocity_modification = max(min(velocity_modification,1.0),-1.0)
                start_action = time.time()
                # if velocity_modification>0:
                #     throttle = velocity_modification
                #     brake = 0.0
                #     self.vehicle.apply_control(carla.VehicleControl(throttle=self.throttle*0.9 + throttle*0.1))
                # else:
                #     throttle = 0.0
                #     brake = velocity_modification
                #     self.vehicle.apply_control(carla.VehicleControl(brake = self.brake*0.9+brake*0.1))
                if brake > throttle:
                    
                    #self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1,  brake = self.brake*0.9+brake*0.1))
                    #self.vehicle.apply_control(carla.VehicleControl(brake = self.brake*0.9+brake*0.1))
                    self.vehicle.apply_control(carla.VehicleControl(brake=self.brake*0.1+brake*0.9))
                    self.throttle = 0.0
                    self.brake = brake
                else:
                    #self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, throttle=self.throttle*0.9 + throttle*0.1))
                    #self.vehicle.apply_control(carla.VehicleControl(throttle=self.throttle*0.9 + throttle*0.1))
                    self.vehicle.apply_control(carla.VehicleControl(throttle=self.throttle*0.1+throttle*0.9))
                    self.throttle = throttle
                    self.brake = 0.0
                end_action = time.time()
                self.previous_steer = steer
                # self.throttle = throttle
                # self.brake = brake
            else:
                steer = self.action_space[action_idx]
                if self.velocity < 20.0:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, throttle=1.0))
                else:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1))
                self.previous_steer = steer
                self.throttle = 1.0
                #self.brake = 0.0
            
            # Traffic Light state
            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)

            self.collision_history = self.collision_obj.collision_data            

            # Rotation of the vehicle in correlation to the map/lane
            self.rotation = self.vehicle.get_transform().rotation.yaw

            # Location of the car
            self.location = self.vehicle.get_location()

            #geo_loc = self.map.transform_to_geolocation(self.location)
            if 0.0< self.velocity <= 10.0:
                color_now = carla.Color(255,0,0,128)
            elif 10.0 < self.velocity < 20.0:
                color_now = carla.Color(0,0,255,128)
            else:
                color_now = carla.Color(0,255,0,128)
            #self.world.debug.draw_point(self.location, size=0.05, life_time=0, color=color_now)

            #transform = self.vehicle.get_transform()
            # Keep track of closest waypoint on the route
            waypoint_index = self.current_waypoint_index
            for _ in range(len(self.route_waypoints)):
                # Check if we passed the next waypoint along the route
                next_waypoint_index = waypoint_index + 1
                wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
                dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],self.vector(self.location - wp.transform.location)[:2])
                if dot > 0.0:
                    waypoint_index += 1
                else:
                    break

            # if self.waypoint_reward_index_75radius != waypoint_index and self.waypoint_reward_index_75radius != waypoint_index+1 and self.waypoint_reward_index_75radius != waypoint_index+2:
            #     self.waypoint_reward_index_75radius = waypoint_index
            #     self.waypoint_reward_75radius = self.route_waypoints[(self.waypoint_reward_index_75radius) % len(self.route_waypoints)]

            # if self.waypoint_reward_index_5radius != waypoint_index and self.waypoint_reward_index_5radius != waypoint_index +1 and self.waypoint_reward_index_5radius != waypoint_index +2: 
            #     self.waypoint_reward_index_5radius = waypoint_index
            #     self.waypoint_reward_5radius = self.route_waypoints[(self.waypoint_reward_index_5radius) % len(self.route_waypoints)]

            # if self.waypoint_reward_index_25radius != waypoint_index and self.waypoint_reward_index_25radius != waypoint_index +1 and self.waypoint_reward_index_25radius != waypoint_index +2:
            #     self.waypoint_reward_index_25radius = waypoint_index
            #     self.waypoint_reward_25radius = self.route_waypoints[(self.waypoint_reward_index_25radius) % len(self.route_waypoints)]

            # if self.waypoint_reward_index_1radius != waypoint_index and self.waypoint_reward_index_1radius != waypoint_index+1 and self.waypoint_reward_index_1radius != waypoint_index+2:
            #     self.waypoint_reward_index_1radius = waypoint_index
            #     self.waypoint_reward_1radius = self.route_waypoints[(self.waypoint_reward_index_1radius) % len(self.route_waypoints)]

            self.current_waypoint_index = waypoint_index
            # Calculate deviation from center of the lane
            self.current_waypoint = self.route_waypoints[ self.current_waypoint_index    % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
            self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),self.vector(self.next_waypoint.transform.location),self.vector(self.location))
            self.center_lane_deviation += self.distance_from_center

            # Get angle difference between closest waypoint and vehicle forward vector
            fwd    = self.vector(self.vehicle.get_velocity())
            wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
            self.angle  = self.angle_diff(fwd, wp_fwd)

             # Update checkpoint for training
            if not self.fresh_start:
                if self.checkpoint_frequency is not None:
                    self.checkpoint_waypoint_index = 0#(self.current_waypoint_index // self.checkpoint_frequency) * self.checkpoint_frequency
            
            # Rewards are given below!
            done = False
            reward = 0
            alpha = 0.01
            if len(self.collision_history) != 0:
                self.reward_velocity=0
                done = True
                reward = -50#50#-15#-7500*alpha+(self.timesteps-1) * alpha
                self.fresh_start = True
            elif self.distance_from_center > self.max_distance_from_center: #2.4
                self.reward_velocity=0
                done = True
                reward = -50#50#-15#-7500*alpha+(self.timesteps-1) * alpha
                self.fresh_start = True
            #elif self.episode_start_time +  < time.time() and self.timesteps > 100 and np.linalg.norm(self.vector(self.location)[0:2]-self.vector(self.spawn_location)[0:2])<3.0:  #ab nun 5 sek
            elif self.timesteps > 500 and np.linalg.norm(self.vector(self.location)[0:2]-self.vector(self.spawn_location)[0:2])<3.0:
                reward = -50#50
                done = True
                self.fresh_start = True 
            # elif self.episode_start_time + 10 < time.time() and self.velocity < 0.0:
            #     reward = -50
            #     self.reward_velocity=0
            #     done = True
            # elif self.velocity > self.max_speed:
            #     reward = -10
            #     self.reward_velocity=0
            #     done = True
            # elif np.linalg.norm(self.vector(self.route_waypoints[0 % len(self.route_waypoints)].transform.location)-self.vector(self.location))<3.0 and self.episode_start_time+8<time.time():#waypoint instead of time
            #     reward = -50
            #     self.reward_velocity=0
            #     done = True


            # Interpolated from 1 when centered to 0 when 3 m from center
            centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
            # # Interpolated from 1 when aligned with the road to 0 when +/- 30 degress of road
            angle_factor = max(1.0 - abs(self.angle / np.deg2rad(30)), 0.0)

            if not done:

                if self.continous_action_space:
                    # if self.velocity < 5.0:
                    #     reward = max(1-np.linalg.norm(self.vector(self.location)[0:2]-self.vector(self.last_waypoint.transform.location)[0:2])/150,0.0)*0.5
                    # else:
                    #     reward = max(1-np.linalg.norm(self.vector(self.location)[0:2]-self.vector(self.last_waypoint.transform.location)[0:2])/150,0.0)
                    #reward ||loc_prev-end||--||loc_curr-end||:
                    # reward += (np.linalg.norm(self.vector(self.previous_location)[0:2]-self.vector(self.last_waypoint.transform.location)[0:2])-np.linalg.norm(self.vector(self.location)[0:2]-self.vector(self.last_waypoint.transform.location)[0:2]))
                    # self.previous_location = self.location
                    #reward 1/||loc_curr-end||
                    # if self.velocity > 0.5:
                    #     reward += 1/np.linalg.norm(self.vector(self.location)[0:2]-self.vector(self.last_waypoint.transform.location)[0:2])
                    # else:
                    #     reward = 0
                    #reward -= alpha
                    #

                    #reward: (||l_prev-l_end||-||l_curr-l_end||)*(d_max-d)
                    lprev_lcurr = np.linalg.norm(self.vector(self.previous_location)[0:2]-self.vector(self.last_waypoint.transform.location)[0:2])-np.linalg.norm(self.vector(self.location)[0:2]-self.vector(self.last_waypoint.transform.location)[0:2])
                    #d_min_dmax = np.linalg.norm(self.vector(self.spawn_location)[0:2]-self.vector(self.last_waypoint.transform.location)[0:2])-np.linalg.norm(self.vector(self.location)[0:2]-self.vector(self.last_waypoint.transform.location)[0:2])
                    d_min_dmax = (2-np.linalg.norm(self.vector(self.location)[0:2]-self.vector(self.last_waypoint.transform.location)[0:2])/np.linalg.norm(self.vector(self.spawn_location)[0:2]-self.vector(self.last_waypoint.transform.location)[0:2]))
                    #t_tmax = 2-self.timesteps/7500
                    t_tmax = (BETA+BETA_TILDE*(1-self.timesteps/TMAX))
                    reward += lprev_lcurr*t_tmax#*(d_min_dmax)#+1)
                    #reward += ((BETA*lprev_lcurr)-BETA_TILDE/TMAX*1/200*self.timesteps)
                    self.previous_location = self.location
                    #if self.velocity > 5.0:
                    #     reward = max(1-np.linalg.norm(self.vector(self.location)[0:2]-self.vector(self.last_waypoint.transform.location)[0:2])/150,0.0)
                    # else:
                    #     reward = 0
                        #reward = max(1-np.linalg.norm(self.vector(self.location)[0:2]-self.vector(self.last_waypoint.transform.location)[0:2])/100,0.0)
                    #print(np.linalg.norm(self.vector(self.current_waypoint.transform.location)-self.vector(self.location)))
                    # if np.linalg.norm(self.vector(self.waypoint_reward_25radius.transform.location)[0:2]-self.vector(self.location)[0:2]) <= 0.25:  
                    #     reward += 1   #+1
                    #     self.waypoint_reward_25radius = self.route_waypoints[(self.waypoint_reward_index_25radius+1) % len(self.route_waypoints)]
                    #     self.waypoint_reward_index_25radius +=1
                    # if np.linalg.norm(self.vector(self.waypoint_reward_5radius.transform.location)[0:2]-self.vector(self.location)[0:2]) <= 0.5 :
                    #     #print('25')
                    #     reward += 1
                    #     self.waypoint_reward_5radius = self.route_waypoints[(self.waypoint_reward_index_5radius+1) % len(self.route_waypoints)]
                    #     self.waypoint_reward_index_5radius +=1
                    # if np.linalg.norm(self.vector(self.waypoint_reward_75radius.transform.location)[0:2]-self.vector(self.location)[0:2]) <= 0.75:
                    #     reward += 1
                    #     #print('375')
                    #     self.waypoint_reward_75radius = self.route_waypoints[(self.waypoint_reward_index_75radius+1) % len(self.route_waypoints)]
                    #     self.waypoint_reward_index_75radius +=1
                    
                    # if np.linalg.norm(self.vector(self.waypoint_reward_1radius.transform.location)[0:2]-self.vector(self.location)[0:2]) <= 0.75:
                    #     reward += 1
                    #     #print('375')
                    #     self.waypoint_reward_1radius = self.route_waypoints[(self.waypoint_reward_index_1radius+1) % len(self.route_waypoints)]
                    #     self.waypoint_reward_index_1radius +=1
                    
                    # reward -= alpha
                    # if self.velocity < self.min_speed:
                    #     reward = (self.velocity / self.min_speed) * centering_factor * angle_factor    
                    # elif self.velocity > self.target_speed:               
                    #     reward = (1.0 - (self.velocity-self.target_speed) / (self.max_speed-self.target_speed)) * centering_factor * angle_factor  
                    # else:                                      
                    #     reward = 0.8*centering_factor * angle_factor + 0.2*self.velocity/self.target_speed#1.0 * centering_factor * angle_factor 
                # else:
                #     reward = 1.0 * centering_factor * angle_factor
            #print(f'reward: {reward}')
            if self.timesteps >= TMAX:
                print('Done over timesteps')
                reward = -50#50#-alpha
                #reward = -10
                done = True
                self.fresh_start = True
            elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
                print('done here')
                done = True
                finished_route = True
                self.fresh_start = True
                reward += 100   #Positive sparse reward to finished the route
                if self.checkpoint_frequency is not None:
                    if self.checkpoint_frequency < self.total_distance//2:
                        self.checkpoint_frequency += 2
                    else:
                        self.checkpoint_frequency = None
                        self.checkpoint_waypoint_index = 0

            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0002)
            while(len(self.rgb_camera.front_rgb) == 0):
                time.sleep(0.0002)
            while(len(self.front_camera.front_front)==0):
                time.sleep(0.0002)
            self.rgb_obs = self.rgb_camera.front_rgb.pop(-1)
            
            
            self.front_obs = self.front_camera.front_front.pop(-1)
            
            self.image_obs = self.camera_obj.front_camera.pop(-1)
            self.image_obs = self.image_obs[0:25, 10:14]
           
            try:
                distance_31 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[1].get_location())[0:2])
            except:
                distance_31 = 200
                
            try:
                distance_4 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[16].get_location())[0:2])
            except:
                distance_4 = 200
                
            try:
                distance_25 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[7].get_location())[0:2])
            except:
                distance_25 = 200
                
            try:
                distance_20 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[9].get_location())[0:2])
            except:
                distance_20 = 200
                
            try:
                distance_ego_front_vehicle = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[5].get_location())[0:2])
            except:
                distance_ego_front_vehicle = 200
                
            if case == 'a':
                if distance_ego_front_vehicle <=42:
                    self.distance.append(distance_ego_front_vehicle)
            if case == 'b' :
                if distance_31<15:
                    self.distance.append(-distance_31)
                if distance_4 <15:
                    self.distance.append(-distance_4)
            if case == 'c':
                if distance_25 <= 42:
                    self.distance.append(distance_25)
                if distance_20 <=42:
                    self.distance.append(distance_20)
                if distance_ego_front_vehicle <= 42:
                    self.distance.append(distance_ego_front_vehicle)
            
            if case == 'd':
                if distance_31<15:
                    self.distance.append(-distance_31)
                if distance_4 <15:
                    self.distance.append(-distance_4)
                if distance_25 <= 42:
                    self.distance.append(distance_25)
                if distance_20 <=42:
                    self.distance.append(distance_20)
                if distance_ego_front_vehicle <= 42:
                    self.distance.append(distance_ego_front_vehicle)

            
                
            # if val == False:
            #     #cv2.imwrite(os.path.join(f'observation_{self.town}_{SAVE_NAME}',f'episode_{episode+1}_rgb',f'obs_img_{self.timesteps}.png'),self.rgb_obs)
            #     #cv2.imwrite(os.path.join(f'observation_{self.town}_{SAVE_NAME}',f'episode_{episode+1}_front',f'obs_img_{self.timesteps}.png'),self.front_obs)
            #     cv2.imwrite(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}',f'episode_{episode+1}',f'obs_img_{self.timesteps}.png'),self.image_obs)
            # else:
            #     #cv2.imwrite(os.path.join(f'observation_{self.town}_{SAVE_NAME}_validation',f'episode_{episode+1}_rgb',f'obs_img_{self.timesteps}.png'),self.rgb_obs)
            #     # cv2.imwrite(os.path.join(f'observation_{self.town}_{SAVE_NAME}_validation',f'episode_{episode+1}_front',f'obs_img_{self.timesteps}.png'),self.front_obs)
            #     cv2.imwrite(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}_validation_{case_name}_{TNAME}_{RUN_NUMBER}',f'episode_{episode+1}_e',f'obs_img_{self.timesteps}.png'),self.image_obs)
            if case != 'e' and len(self.distance) > 0:
                self.image_obs = self.disappear_object(self.image_obs, self.distance)

            # if val == False:
            #     #cv2.imwrite(os.path.join(f'observation_{self.town}_{SAVE_NAME}',f'episode_{episode+1}_rgb',f'obs_img_{self.timesteps}.png'),self.rgb_obs)
            #     #cv2.imwrite(os.path.join(f'observation_{self.town}_{SAVE_NAME}',f'episode_{episode+1}_front',f'obs_img_{self.timesteps}.png'),self.front_obs)
            #     cv2.imwrite(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}',f'episode_{episode+1}',f'obs_img_{self.timesteps}.png'),self.image_obs)
            # else:
            #     #cv2.imwrite(os.path.join(f'observation_{self.town}_{SAVE_NAME}_validation',f'episode_{episode+1}_rgb',f'obs_img_{self.timesteps}.png'),self.rgb_obs)
            #     # cv2.imwrite(os.path.join(f'observation_{self.town}_{SAVE_NAME}_validation',f'episode_{episode+1}_front',f'obs_img_{self.timesteps}.png'),self.front_obs)
            #     cv2.imwrite(os.path.join(SAVE_PATH,f'observation_{self.town}_{SAVE_NAME}_validation_{case_name}_{TNAME}_{RUN_NUMBER}',f'episode_{episode+1}',f'obs_img_{self.timesteps}.png'),self.image_obs)
            #distance_ego_front_vehicle = np.linalg.norm(self.vector(self.location)[0:2]-self.vector(self.actor_list[5].get_location())[0:2])
            
            self.image_obs = cv2.cvtColor(self.image_obs, cv2.COLOR_RGB2GRAY)
            self.image_obs = self.image_obs.flatten()
            normalized_velocity = self.velocity/self.target_speed
            normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
            normalized_angle = abs(self.angle / np.deg2rad(20))
            ########Uncertainty
            a = 0
            b = 0
            c = 0
            d = 0
            e = 0
            if case == 'a':
                a = 1
            elif case == 'b':
                b = 1
            elif case == 'c':
                c = 1
            elif case == 'd':
                d = 1
            elif case == 'e':
                e = 1
            self.navigation_obs = np.array([self.throttle, self.velocity, normalized_velocity, normalized_distance_from_center, normalized_angle, self.brake, a, b, c, d])
            
            # Remove everything that has been spawned in the env
            if done:
                self.center_lane_deviation = self.center_lane_deviation / self.timesteps
                self.distance_covered = abs(self.current_waypoint_index - self.checkpoint_waypoint_index)
                self.reward_velocity=0
                for sensor in self.sensor_list:
                    sensor.destroy()
                
                self.remove_sensors()
                print('destroy actor')
                # for actor in self.actor_list:
                #     try:
                #         actor.destroy()
                #     except:
                #         pass
           # info_map = self.map.save_to_disk('Town2_xodr')
           
            #xodr_map.write(info_map)
            # time.sleep(2)
            if reward < -50:
                reward = -50 #There is some Carla bug for the initial spawning, that the ego vehicle is spawnd twice
            action_time = end_action-start_action
            if done and finished_route == False:
                lprev_lcurr=0
                t_tmax = 1
                d_min_dmax = 1
            #print(self.vector(self.actor_list[5].get_location())[0:2])
            try:
                front_vehicle_location = self.actor_list[5].get_location()
            except:
                front_vehicle_location = self.actor_list[0].get_location()
            return [self.image_obs, self.navigation_obs], reward, done, [self.distance_covered, self.center_lane_deviation] , self.location, self.velocity, action_time, self.brake, np.linalg.norm(self.vector(self.current_waypoint.transform.location)[0:2]-self.vector(self.location)[0:2]), self.throttle, action_idx, lprev_lcurr, t_tmax,d_min_dmax, distance_ego_front_vehicle,front_vehicle_location, self.velocity_front, finished_route#, self.vector(self.waypoint_reward_25radius.transform.location)[0:2], self.vector(self.location)[0:2], self.waypoint_reward_index_25radius, self.waypoint_reward_index_5radius, self.waypoint_reward_index_75radius, waypoint_index

        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            self.reward_velocity=0
            if self.display_on:
                pygame.quit()

    # def string(self, location, episode):
    #     print('Start writing')
    #     print(location)
    #     self.world.debug.draw_string(location, str(episode), False, carla.Color(238, 18, 137, 128), 0)

# -------------------------------------------------
# Creating and Spawning Pedestrians in our world |
# -------------------------------------------------

    # Walkers are to be included in the simulation yet!
    def create_pedestrians(self):
        try:

            # Our code for this method has been broken into 3 sections.

            # 1. Getting the available spawn points in  our world.
            # Random Spawn locations for the walker
            walker_spawn_points = []
            for i in range(NUMBER_OF_PEDESTRIAN):
                spawn_point_ = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point_.location = loc
                    walker_spawn_points.append(spawn_point_)

            # 2. We spawn the walker actor and ai controller
            # Also set their respective attributes
            for spawn_point_ in walker_spawn_points:
                walker_bp = random.choice(
                    self.blueprint_library.filter('walker.pedestrian.*'))
                walker_controller_bp = self.blueprint_library.find(
                    'controller.ai.walker')
                # Walkers are made visible in the simulation
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # They're all walking not running on their recommended speed
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute(
                        'speed', (walker_bp.get_attribute('speed').recommended_values[1]))
                else:
                    walker_bp.set_attribute('speed', 0.0)
                walker = self.world.try_spawn_actor(walker_bp, spawn_point_)
                if walker is not None:
                    walker_controller = self.world.spawn_actor(
                        walker_controller_bp, carla.Transform(), walker)
                    self.walker_list.append(walker_controller.id)
                    self.walker_list.append(walker.id)
            all_actors = self.world.get_actors(self.walker_list)

            # set how many pedestrians can cross the road
            #self.world.set_pedestrians_cross_factor(0.0)
            # 3. Starting the motion of our pedestrians
            for i in range(0, len(self.walker_list), 2):
                # start walker
                all_actors[i].start()
            # set walk to random point
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())

        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])


# ---------------------------------------------------
# Creating and Spawning other vehciles in our world|
# ---------------------------------------------------


    def set_other_vehicles(self):
        try:
            # NPC vehicles generated and set to autopilot
            # One simple for loop for creating x number of vehicles and spawing them into the world
            if len(self.spawn_points) == 0:
                start = True
            #print(len(self.map.get_spawn_points()))
            # spawning_case = random.randint(0,4)   #[first front vehicle, backward vehicle, all front vehicles, all vehicles, oracle]
            # print(spawning_case)
            # spawn_list_cases =[[31,30,28,87,26,25,23,20,19,97,96,22,21,5,4,18,17],[30,28,87,27,26,25,23,20,19,97,96,22,21,5,18,17],[31,28,87,26,23,19,96,21,5,4,18,17],[30,28,87,26,23,19,96,21,5,18,17],[31, 30,28,87, 27, 26, 25, 23, 20, 19, 97, 96, 22, 21, 5, 4, 18, 17]]
            spawn_list = [31, 30,28,87, 27, 26, 25, 23, 20, 19, 97, 96, 22, 21, 5, 4, 18, 17]
           # print(spawn_list)
            j = 0 #For Town02
            #j=11 #For Town04
            tm = self.client.get_trafficmanager(8000)
            tm_port = tm.get_port()
            #self.spawn_points = []
            
            if len(self.spawn_points) == 0:
                for _ in range(0, NUMBER_OF_VEHICLES):
                    #spawn_point = random.choice(self.map.get_spawn_points())
                    
                    bp_vehicle = random.choice(self.blueprint_library.filter('vehicle'))
                    
                    if bp_vehicle.id == 'vehicle.tesla.cybertruck' or bp_vehicle.id == 'vehicle.volkswagen.t2' or bp_vehicle.id == 'vehicle.kawasaki.ninja' or bp_vehicle.id == 'vehicle.micro.microlino' or bp_vehicle.id == 'vehicle.mitsubishi.fusorosa' or bp_vehicle.id == 'vehicle.bh.crossbike' or bp_vehicle.id == 'vehicle.diamondback.century' or bp_vehicle.id == 'vehicle.gazelle.omafiets' or bp_vehicle.id == 'vehicle.volkswagen.t2_2021' or bp_vehicle.id == 'vehicle.mercedes.sprinter' or bp_vehicle.id == 'vehicle.ford.ambulance' or bp_vehicle.id == 'vehicle.carlamotors.firetruck' or bp_vehicle.id == 'vehicle.carlamotors.european_hgv' or bp_vehicle.id == 'vehicle.carlamotors.carlacola' or bp_vehicle.id == 'vehicle.harley-davidson.low_rider' or bp_vehicle.id == 'vehicle.vespa.zx125' or bp_vehicle.id == 'vehicle.yamaha.yzf':
                        continue
                    try:
                        spawn_point = self.map.get_spawn_points()[spawn_list[j]]
                        j+=1
                    except:
                        spawn_point = random.choice(self.map.get_spawn_points())
                    
                    other_vehicle = self.world.try_spawn_actor(
                        bp_vehicle, spawn_point)
                    if other_vehicle is not None:
                        other_vehicle.set_autopilot(True,tm_port)
                        danger_car = other_vehicle
                        
                        self.actor_list.append(other_vehicle)
                        self.spawn_points.append(spawn_point)
                        self.other_vehicles.append(bp_vehicle)
                    #j+=1
                #tm.global_distance_to_leading_vehicle(2.5)
                tm.global_percentage_speed_difference(75) #80
            else:
                for i in range(0,len(self.spawn_points)):
                    other_vehicle = self.world.try_spawn_actor(
                        self.other_vehicles[i], self.spawn_points[i])
                    if other_vehicle is not None:
                        #print(f'New actor is spawned: {other_vehicle}')
                        other_vehicle.set_autopilot(True, tm_port)
                        self.actor_list.append(other_vehicle)
                tm.global_percentage_speed_difference(75)  #80
            print(f'Number of NPCs: {len(self.actor_list)}') 
            start = False   
            #print("NPC vehicles have been generated in autopilot mode.")
        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list])
        #return spawning_case

# ----------------------------------------------------------------
# Extra very important methods: their names explain their purpose|
# ----------------------------------------------------------------

    # Setter for changing the town on the server.
    def change_town(self, new_town):
        self.world = self.client.load_world(new_town)


    # Getter for fetching the current state of the world that simulator is in.
    def get_world(self) -> object:
        return self.world


    # Getter for fetching blueprint library of the simulator.
    def get_blueprint_library(self) -> object:
        return self.world.get_blueprint_library()


    # Action space of our vehicle. It can make eight unique actions.
    # Continuous actions are broken into discrete here!
    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi: angle -= 2 * np.pi
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle


    def distance_to_line(self, A, B, p):
        num   = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom


    def vector(self, v):
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])


    def get_discrete_action_space(self):
        action_space = \
            np.array([
            -0.50,
            -0.30,
            -0.10,
            0.0,
            0.10,
            0.30,
            0.50
            ])
        return action_space

    # Main vehicle blueprint method
    # It picks a random color for the vehicle everytime this method is called
    def get_vehicle(self, vehicle_name):
        blueprint = self.blueprint_library.filter(vehicle_name)[0]
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint


    # Spawn the vehicle in the environment
    def set_vehicle(self, vehicle_bp, spawn_points):
        # Main vehicle spawned into the env
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)


    # Clean up method
    def remove_sensors(self):
        self.camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.env_camera_obj = None
        self.front_camera = None
        self.collision_history = None
        self.wrong_maneuver = None

    def save_location(self, location_list):
        '''
        written by Natalie Grabowsky
        '''
        location_file = open('geo_location.txt', 'a')
        print(f'Location list: {location_list}')
        location_file.write(self.map.transform_to_geolocation(location_list))
            
#-------------------------------------------------------------------------------
# Create perturbation
#-------------------------------------------------------------------------------
    def disappear_object(self, img, distance):
        '''
        Find the front vehicle and vanish it.
        Inputparameter:
        -----------------------------------------
        img       : Array: Image in semantic segmnetation colors of CARLA
        distance  : List: Distance between ego vehilce and vehicles along the ego line
        color     : Tuple: Color of class car
        fill_color: Tuple: Color of class road

        Outputparameter:
        ------------------------------------------
        img       : Array: Image in semantic segmentation colors of CARLA without the vehicle in front of us
        '''
        color = [142,0,0]
        fill_color = [128,64,128] 
        ego_position = [20,2]
        img = copy.deepcopy(img)
        for dis in distance:
            pixel_height = int(dis/2)
            
            vehicle_position = [ego_position[0]-pixel_height, ego_position[1]]
            if vehicle_position[0]>np.shape(img)[0]-1:
                vehicle_position[0] = np.shape(img)[0]-1
            if vehicle_position[0] < 0:
                vehicle_position[0] = 1
            if np.all(img[vehicle_position[0],vehicle_position[1]] == color):
                
                i = -1
                j = 1
                y = vehicle_position[0]
                x = vehicle_position[1]
                colorize = True
               
               
                img[y,x] = np.array(fill_color)
                
                while colorize == True:
                    if y+i<0:
                        i = 0
                    if y+i<=19:
                        colorize == False
                    if y+j>=19:
                        colorize == False
                    if np.all(img[y+i,x,:] == color):
                        img[y+i,x] = np.array(fill_color)
                        i-=1

                    elif y+j < np.shape(img)[0] and np.all(img[y+j,x,:] == color):
                        img[y+j,x] = np.array(fill_color)
                        j+=1
                    else:
                        colorize = False
            elif np.all(img[vehicle_position[0]-1,vehicle_position[1]] == color):
                i = -1
                j = 1
                y = vehicle_position[0]
                x = vehicle_position[1]
                colorize = True
                img[y,x] = np.array(fill_color)
                while colorize == True:
                    if y+i<=19:
                        colorize == False
                    if y+j>=19:
                        colorize == False
                    if np.all(img[y+i,x,:] == color):
                        img[y+i,x] = np.array(fill_color) 
                        i-=1
                    elif y+j < np.shape(img)[0] and np.all(img[y+j,x,:] == color):
                        img[y+j,x] = np.array(fill_color)
                        j+=1
                    else:
                        colorize = False
            img[19:22,2] = np.array(color)
        return img

#if spawning_case == 4:
#     try:
#         distance_31 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[1].get_location())[0:2])
#     except:
#         distance_31 = 200
#         print('No NPC on spawning point 31')
#     try:
#         distance_4 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[16].get_location())[0:2])
#     except:
#         distance_4 = 200
#         print('No NPC on spawning point 4')
#     try:
#         distance_25 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[7].get_location())[0:2])
#     except:
#         distance_25 = 200
#         print('No NPC on spawning point 25')
#     try:
#         distance_20 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[9].get_location())[0:2])
#     except:
#         distance_20 = 200
#         print('No NPC on spawning point 20')
#     try:
#         distance_ego_front_vehicle = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[5].get_location())[0:2])
#     except:
#         distance_ego_front_vehicle = 200
#         print('No NPC on spawning point 27')
#     if case == 'a':
#         if distance_ego_front_vehicle <=42:
#             self.distance.append(distance_ego_front_vehicle)
#     if case == 'b' :
#         if distance_31<15:
#             self.distance.append(-distance_31)
#         if distance_4 <15:
#             self.distance.append(-distance_4)
#     if case == 'c':
#         if distance_25 <= 42:
#             self.distance.append(distance_25)
#         if distance_20 <=42:
#             self.distance.append(distance_20)
#         if distance_ego_front_vehicle <= 42:
#             self.distance.append(distance_ego_front_vehicle)
    
#     if case == 'd':
#         if distance_31<15:
#             self.distance.append(-distance_31)
#         if distance_4 <15:
#             self.distance.append(-distance_4)
#         if distance_25 <= 42:
#             self.distance.append(distance_25)
#         if distance_20 <=42:
#             self.distance.append(distance_20)
#         if distance_ego_front_vehicle <= 42:
#             self.distance.append(distance_ego_front_vehicle)

# elif spawning_case == 0:
#     try:
#         distance_31 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[1].get_location())[0:2])
#     except:
#         distance_31 = 200
#         print('No NPC on spawning point 31')
#     try:
#         distance_4 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[15].get_location())[0:2])
#     except:
#         distance_4 = 200
#         print('No NPC on spawning point 4')
#     try:
#         distance_25 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[6].get_location())[0:2])
#     except:
#         distance_25 = 200
#         print('No NPC on spawning point 25')
#     try:
#         distance_ego_front_vehicle = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[8].get_location())[0:2])
#     except:
#         distance_ego_front_vehicle = 200
#         print('No NPC on spawning point 20')

#     if case == 'a':
#         if distance_ego_front_vehicle <=42:
#             self.distance.append(distance_ego_front_vehicle)
#     if case == 'b' :
#         if distance_31<15:
#             self.distance.append(-distance_31)
#         if distance_4 <15:
#             self.distance.append(-distance_4)
#     if case == 'c':
#         if distance_25 <= 42:
#             self.distance.append(distance_25)
#         if distance_ego_front_vehicle <= 42:
#             self.distance.append(distance_ego_front_vehicle)
    
#     if case == 'd':
#         if distance_31<15:
#             self.distance.append(-distance_31)
#         if distance_4 <15:
#             self.distance.append(-distance_4)
#         if distance_25 <= 42:
#             self.distance.append(distance_25)
#         if distance_ego_front_vehicle <= 42:
#             self.distance.append(distance_ego_front_vehicle)

# elif spawning_case == 1:
#     try:
#         distance_25 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[6].get_location())[0:2])
#     except:
#         distance_25 = 200
#         print('No NPC on spawning point 25')
#     try:
#         distance_20 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[8].get_location())[0:2])
#     except:
#         distance_20 = 200
#         print('No NPC on spawning point 20')
#     try:
#         distance_ego_front_vehicle = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[4].get_location())[0:2])
#     except:
#         distance_ego_front_vehicle = 200
#         print('No NPC on spawning point 27')
#     if case == 'a':
#         if distance_ego_front_vehicle <=42:
#             self.distance.append(distance_ego_front_vehicle)
#     # if case == 'b':
#     #     continue

#     if case == 'c':
#         if distance_25 <= 42:
#             self.distance.append(distance_25)
#         if distance_20 <=42:
#             self.distance.append(distance_20)
#         if distance_ego_front_vehicle <= 42:
#             self.distance.append(distance_ego_front_vehicle)
    
#     if case == 'd':
        
#         if distance_25 <= 42:
#             self.distance.append(distance_25)
#         if distance_20 <=42:
#             self.distance.append(distance_20)
#         if distance_ego_front_vehicle <= 42:
#             self.distance.append(distance_ego_front_vehicle)

# elif spawning_case == 2:
#     try:
#         distance_31 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[1].get_location())[0:2])
#     except:
#         distance_31 = 200
#         print('No NPC on spawning point 31')
#     try:
#         distance_4 = np.linalg.norm(self.vector(self.vehicle.get_location())[0:2]-self.vector(self.actor_list[11].get_location())[0:2])
#     except:
#         distance_4 = 200
#         print('No NPC on spawning point 4')
    
#     distance_ego_front_vehicle = 200
#     # if case == 'a':
#     #     continue
#     if case == 'b' :
#         if distance_31<15:
#             self.distance.append(-distance_31)
#         if distance_4 <15:
#             self.distance.append(-distance_4)
#     # if case == 'c':
#     #     continue
    
#     if case == 'd':
#         if distance_31<15:
#             self.distance.append(-distance_31)
#         if distance_4 <15:
#             self.distance.append(-distance_4)
    
