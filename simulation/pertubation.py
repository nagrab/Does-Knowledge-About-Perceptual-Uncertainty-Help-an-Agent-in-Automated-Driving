import numpy as np
import cv2
import os
import random
import copy
import sys
'''
Shifting vehicles in semantic segmentation to burning the truth semseg image.
Idea: Compute contours of the searched objects, shift contour, overwrite oracle
contours and then draw shifted contours. 

Written by Natalie Grabowsky
'''
def scale_contour(contour, scale):
    '''
    Based on: https://medium.com/analytics-vidhya/tutorial-how-to-scale-and-rotate-contours-in-opencv-using-python-f48be59c35a2
    '''
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = contour - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

def cart2pol(x, y):
    '''
    Based on: https://medium.com/analytics-vidhya/tutorial-how-to-scale-and-rotate-contours-in-opencv-using-python-f48be59c35a2
    '''
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    '''
    Based on: https://medium.com/analytics-vidhya/tutorial-how-to-scale-and-rotate-contours-in-opencv-using-python-f48be59c35a2
    '''
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def rotate_contour(cnt, angle):
    '''
    Based on: https://medium.com/analytics-vidhya/tutorial-how-to-scale-and-rotate-contours-in-opencv-using-python-f48be59c35a2
    '''
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    
    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    
    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)
    
    xs, ys = pol2cart(thetas, rhos)
    
    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated

def reshape_objects(oracle_image, color, ego_position):
    '''
    Reshape objects, like increasing or decreasing the object, or rotation it 90 degree

    Inputparameter:
    -------------------------------------------
    Oracle_image : Array: Array of the given image
    Color        : Array: Array of rgb values to find the object color (we are in the setting of semantic segmentation)
    Ego position : Array: Tuple of x and y for center of ego vehicle
    
    Outputparameter:
    ------------------------------------------
    Shifted_image: Array: Array of reshaped/rotated image
    '''
    contours,_ = find_contours(oracle_image, color)
    contours_copy = copy.deepcopy(contours)
    filled_image = fill_oracle_contours(contours_copy, oracle_image)
    for i,contour in enumerate(contours):
        if np.min(contour[:,0,0])<ego_position[0] and np.max(contour[:,0,0])>ego_position[0] and np.min(contour[:,0,1])<ego_position[1] and np.max(contour[:,0,1])> ego_position[1]:
            j = i
            continue
        else:
            reshape   = bool(np.random.randint(2))
            rotate    = bool(np.random.randint(2))
        if reshape == True:
            scale = np.random.uniform(0.1,4)
            try:
                contour = scale_contour(copy.deepcopy(contour), scale)
                list_contours = list(contours)
                list_contours[i] = contour
                contours = tuple(list_contours)
                
            except:
                print('scaling does not work for this contour')
        if rotate == True:
            angle = np.random.uniform(0,90)
            try:
                contour  = rotate_contour(copy.deepcopy(contour), angle)
                list_contours = list(contours)
                list_contours[i] = contour
                contours = tuple(list_contours)
                
            except:
                print('Rotation does not work for this contour.')

    shifted_image = cv2.fillPoly(filled_image, pts=contours, color=color)
    return shifted_image
    # shifted_image_gt = cv2.fillPoly(filled_image_gt, pts=contours, color=color_id)
    # cv2.imwrite(os.path.join(target_path,file+'_rero_semantic_cs.png'),shifted_image )
    # cv2.imwrite(os.path.join(target_path,file+'_rero_semantic.png'), shifted_image_gt)

        
def cloning_objects(oracle_image, color,color_road,ego_position):
    '''
    Cloning objects to have more vehicles on the road, which are not really there (predict more then expected)
    Inputparameter:
    -------------------------------------------
    Oracle_image: Array: Array of the given image
    Color       : Array: Array of rgb values to find the object color (we are in the setting of semantic segmentation)
    Color_road  : Array: Array of rgb values of the class road to find the location of roads
    Ego_position: Tuple: Position of ego vehicle
    '''
    #-> get contours of vehicle x-> get contours of road x-> shifting it -> save as a new image
    contours, masked_img= find_contours(oracle_image, color)
    contours_road , masked_img_road = find_contours(oracle_image, color_road)
    contours_clone = []
    contours_copy = copy.deepcopy(contours)
    list_contours = list(contours_copy)
    np.random.shuffle(list_contours)
    contours_clone = compute_shift(contours_copy, oracle_img, ego_position,clone = True)
    # for contour_road in contours_road:
    #     if cv2.contourArea(contour_road)<10:
    #         continue
    #     for j,contour in enumerate(list_contours):
    #         if cv2.contourArea(contour_road)> cv2.contourArea(contour):
               
    #             min_x_road  = np.min(contour_road[:,0,0])
    #             max_x_road  = np.max(contour_road[:,0,0])
    #             min_y_road  = np.min(contour_road[:,0,1])
    #             max_y_road  = np.max(contour_road[:,0,1])
    #             min_x       = np.min(contour[:,0,0])
    #             max_x       = np.max(contour[:,0,0])
    #             min_y       = np.min(contour[:,0,1])
    #             max_y       = np.max(contour[:,0,1])
    #             dist_x_road = abs(min_x_road-max_x_road)
    #             dist_y_road = abs(min_y_road-max_y_road)
    #             dist_x      = abs(min_x-max_x)
    #             dist_y      = abs(min_y-max_y)
    #             if dist_x <= dist_x_road and dist_y < dist_y_road:
    #                 difference_x = dist_x_road-dist_x
    #                 difference_y = dist_y_road-dist_y
    #                 x = random.random()
    #                 y = random.random()
    #                 x_shift = min_x_road+x*difference_x
    #                 y_shift = min_y_road+y*difference_y 
    #                 contour[:,0,0] = contour[:,0,0]-min_x
    #                 contour[:,0,1] = contour[:,0,1]-min_y
    #                 contour[:,0,0] = contour[:,0,0]+x_shift+min_x_road
    #                 contour[:,0,1] = contour[:,0,1]+y_shift+min_y_road
    #                 contours_clone.append(contour)
    #                 list_contours.pop(j)

    #                 break
    contours_clone =tuple(contours_clone)+contours
    shifted_image = cv2.fillPoly(oracle_image, pts=contours_clone, color=color)
    return shifted_image
    # shifted_gt  = cv2.fillPoly(gt_image,pts = contours_clone, color = id )
    # cv2.imwrite(os.path.join(target_path,file+'_cloned_semantic_cs.png'),shifted_image )
    # cv2.imwrite(os.path.join(target_path, file+'_cloned_semantic.png'), shifted_gt)

def disappear_objects(oracle_image,  color=(), color_id=14, ego_position=(0,0), ids = None, instance = np.array(0)):
    '''
    Objects disappeard in an image.
    Inputparameter:
    ------------------------------------
    Oracle_image: Array: Image where objects should be disappeared
    Color       : Array: Array of color of the objects, which should be disappear
    Color_id    : Int  : Id of color
    Ego_position: Tuple: Position of ego vehicle
    Ids         : List : List of ids (str) for the instance segmentation
    Instance    : Array: instance segmentation image

    Outputparameter:
    ------------------------------------
    Ids_update  : List : List of new ids
    Image       : Array: Disappeard image
    '''
    contours, masked_img = find_contours(oracle_image, color)
    #instance = cv2.imread(f'{source_path}/{file}_instance.png')
    if len(contours)<=1:
        print('Only ego vehicle found. We do not want to disapear it.')
        return []
    
   
    ids_new = []
    ids_update = []
    non_ids = []
    union_ids = []
    contours_without_ego = []
    #Delete the contours which are too small or ego vehicle
    for k,contour in enumerate(contours):
        if np.min(contour[:,0,0])<ego_position[0] and np.max(contour[:,0,0])>ego_position[0] and np.min(contour[:,0,1])<ego_position[1] and np.max(contour[:,0,1])> ego_position[1]:
            if len(contours_without_ego)==0:
                list_cont = list(contours)
                list_cont.pop(k)
                contours_without_ego = tuple(list_cont)
            else:
                list_cont = list(contours_without_ego)
                for i, con in enumerate(list_cont):
                    if np.all(contour == con):
                        idx_del = i
                        break
                list_cont.pop(i)
                contours_without_ego = tuple(list_cont)
            continue
        elif cv2.contourArea(contour)<=8:
            if len(contours_without_ego)==0:
                list_cont = list(contours)
                list_cont.pop(k)
                contours_without_ego = tuple(list_cont)
            else:
                list_cont = list(contours_without_ego)
                for i, con in enumerate(list_cont):
                    if np.all(contour == con):
                        idx_del = i
                        break
                list_cont.pop(i)
                contours_without_ego = tuple(list_cont)
            continue
        color_id_array = instance[contour[0,0,1], contour[0,0,0]]

        ids_new.append(f'{color_id_array[1]}-{color_id_array[0]}')
    #Compute the disappearing contours
    contours = contours_without_ego
    if len(contours) == 0:
        print('No disappearing object found.')
        return[]
    number_disappear = np.random.randint(len(contours)+1) #Theoretically here is also the ego vehicle, which we do not want to remove
    if number_disappear == 0:
        number_disappear = 1
    
    if len(ids)!=0:
        for id in ids_new:
            if id in ids:
                union_ids.append(id)
            else:
                non_ids.append(id) 
    else:
        non_ids = ids_new
    ids_update = union_ids
    while number_disappear > len(ids_update):
        number_idx = np.random.randint(len(non_ids))
        number = non_ids[number_idx]
        ids_update.append(number)
        non_ids.pop(number_idx)
    for id  in non_ids:
        for k, contour in enumerate(contours):
            if (instance[contour[0,0,1], contour[0,0,0]][0],instance[contour[0,0,1], contour[0,0,0]][1],instance[contour[0,0,1], contour[0,0,0]][2])==(int(id.split('-')[1]),int(id.split('-')[0]),color_id):
                list_cont = list(contours)
                list_cont.pop(k)
                contours = tuple(list_cont)
                break
       
    oracle_image = fill_oracle_contours(contours, oracle_image)
    # cv2.imwrite(os.path.join(target_path, file+'_disappear_semantic_cs.png'), oracle_image)
    # cv2.imwrite(os.path.join(target_path, file+'_disappear_semantic.png'), gt_image)
    return ids_update, oracle_image
        



def fill_oracle_contours(contours, oracle_img):
    '''
    Fill the oracle contours with a suitable color and delete the objects.

    Inputparameter:
    ---------------------------------------
    Contours  : Array: Array of lists return the edges of the contour polygons
    Oracle_img: Array: Input image where we want to find the objects

    Outputparameter:
    --------------------------------------
    Oracle_img: Array: Output with deleted objects
    '''
    for contour in contours:
        min_x = np.min(contour[:,0,0])
        max_x = np.max(contour[:,0,0])
        min_y = np.min(contour[:,0,1])
        max_y = np.max(contour[:,0,1])
        if min_x < 0:
            min_x = 0
        if max_x > np.shape(oracle_img)[1]-1:
            max_x = np.shape(oracle_img)[1]-1
        if min_y < 0:
            min_y = 0
        if max_y > np.shape(oracle_img)[0]-1:
            max_y = np.shape(oracle_img)[0]-1
        dist_x = abs(max_x-min_x)
        dist_y = abs(max_y-min_y)
        if dist_x > dist_y:
            if max_x == np.shape(oracle_img)[1]-1:
                    max_x -=1
            while min_y <= max_y:
                
                for x in range(min_x, max_x+1):
                    oracle_img[max_y,x] = oracle_img[max_y,x-1]
                    oracle_img[min_y,x] = oracle_img[min_y,x-1]
                
                min_y += 1
                max_y -= 1                
        elif dist_x <= dist_y:
            if max_y == np.shape(oracle_img)[0]-1:
                max_y -=1
            while min_x <=max_x:
                for y in range(min_y, max_y+1):
                    oracle_img[y,max_x] = oracle_img[y-1, max_x]
                    oracle_img[y,min_x] = oracle_img[y-1, min_x]
                  
                min_x += 1
                max_x -= 1
                    
    return oracle_img


def compute_shift(contours_shift, oracle_img, ego_position,clone = True):
    '''
    Compute Shift via multivariat Gaussian distribution for each object. 
    The script will compute first a mean for x and y shifting, and then it compute the true shifting, via analysing the shape of image.

    Inputparameter:
    ----------------------------------------------------------
    Contours_shift: Array: Array of lists return the edges of the contour polygons
    Oracle_img    : Array: Input image where we want to find the contours
    Ego position  : Array: Tuple of x and y for center of ego vehicle
    Clone         : Bool : If true the shift will larger than 1 else only shifting

    Outputparameter:
    ----------------------------------------------------------
    Contours_shift: Array: Array of lists return the updated/shifted edges of the contour polygons
    '''
    for contour in contours_shift:
        if np.min(contour[:,0,0])<ego_position[0] and np.max(contour[:,0,0])>ego_position[0] and np.min(contour[:,0,1])<ego_position[1] and np.max(contour[:,0,1])> ego_position[1]:
            continue
        shift = 1
        if clone == True:
            shift = np.random.uniform(-5,5)
            if shift > -1 and shift <1:
                shift = 0
        shift = shift*np.random.normal(size = 2)
        image_shape = np.shape(oracle_img)
        x_shift_total = shift[1]*(abs(np.max(contour[:,0,0])-np.min(contour[:,0,0])))
        y_shift_total = shift[0]*(abs(np.max(contour[:,0,1])-np.min(contour[:,0,1])))
        contour[:,0,0]= contour[:,0,0]+x_shift_total
        contour[:,0,1]= contour[:,0,1]+y_shift_total

    return contours_shift



def find_contours(oracle_img, color):
    '''
    Find contours for the objects which we want to shift.

    Inputparameter:
    ----------------------------------------------------
    Oracle_img: Array: Input image where we want to find the contours
    Color     : Array: Array of rgb values to find the object color (we are in the setting of semantic segmenation)

    Outputparameter:
    ----------------------------------------------------
    Contours  : Array: Array of lists return the edges of the contour polygons
    Masked_img: Array: Binary mask with contours
    '''
    hsv_img = cv2.cvtColor(oracle_img, cv2.COLOR_BGR2HSV)
    white_img = np.array(np.ones((np.shape(oracle_img)[0], np.shape(oracle_img)[1],3))*255,np.uint8)
    color_hsv = cv2.cvtColor(np.array(np.reshape(color, (1,1,np.shape(color)[0])),np.uint8), cv2.COLOR_BGR2HSV)
    lower_bound = color_hsv[0][0]   
    upper_bound = lower_bound  
    masked = cv2.inRange(hsv_img, lower_bound, upper_bound)
    masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    masked_img = cv2.bitwise_and(white_img, masked )
    contours, _ = cv2.findContours(cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours, masked_img



def shifting_objects(oracle_image,  color, ego_position):
    '''
    Main script for shifting object in a image.

    Inputparameter:
    -------------------------------------------
    Oracle_image:  Array: Array of the given image
    Color        : Array: Array of rgb values to find the object color (we are in the setting of semantic segmentation)
    Ego position : Array: Tuple of x and y for center of ego vehicle
    
    Outputparameter:
    ------------------------------------------
    Shifted_image: Array: Array of shifted object image
    '''
    contours,_ = find_contours(oracle_image, color)
    contours_shift = copy.deepcopy(contours)
    shift = compute_shift(contours_shift, oracle_image, ego_position)
    image_without_objects = fill_oracle_contours(contours, oracle_image)
    shifted_image = cv2.fillPoly(image_without_objects, pts=shift, color=color)
    return shifted_image
    # shifted_image_gt = cv2.fillPoly(image_without_objects_gt, pts = shift, color=id)
    # cv2.imwrite(os.path.join(target_path,file+'_shifting_semantic_cs.png'),shifted_image)
    # cv2.imwrite(os.path.join(target_path,file+'_shifting_semantic.png'), shifted_image_gt)


# def main(source_path, target_path, color, color_id, color_road, color_road_id, ego_position, time_line_dis, gt_path):
#     '''
#     Main script. Here we will choose the kind of transformation
#     Inputparameter:
#     -------------------------------------------
#     Source_path    : Str  : Path from the input image where the objects should be shifting
#     Target_path    : Str  : Path where the shifting images should be saved 
#     Color          : Array: Array of rgb values to find the object color (we are in the setting of semantic segmentation)
#     Color_id       : Int  : Id of color
#     Color_road     : Array: Array of rgb values of class road
#     Color_road_id  : Int  : Id of color road
#     Ego position   : Array: Tuple of x and y for center of ego vehicle
#     Time_line_dis  : Int  : How many images should be the object disappeared
#     Gt_path        : Str  : Path where the ground truth images lie
#     '''
#     tld         = 0
#     tlc         = 0
#     ids         = []
#     ids_old     = []
#     number_disp = []
#     disap = True
#     if gt_path != source_path: 
#         print('Please add also a target gt path, to use the same dataloader later.')
#         sys.exit()

#     for root, dir, files in os.walk(source_path):
#         print(source_path)
#         print(root)
#         for file in sorted(files):
#             if file.endswith('semantic_cs.png'):
#                 oracle_image = cv2.imread(os.path.join(source_path,file))
#                 gt_image = cv2.imread(os.path.join(gt_path, file.split('_')[0]+'_'+file.split('_')[1]+'.png'))[:,:,-1]
#                 # shifting = bool(np.random.randint(2))
#                 # cloning  = bool(np.random.randint(2))
#                 # reshape  = bool(np.random.randint(2))
#                 shifting = True
#                 cloning  = True
#                 reshape  = True
#                 if cloning == True:
#                     cloning_objects(copy.deepcopy(oracle_image), file.split('_')[0], target_path, color, ego_position, color_road, copy.deepcopy(gt_image), color_id, color_road_id)
#                 if shifting == True:
#                     shifting_objects(copy.deepcopy(oracle_image), file.split('_')[0], target_path, color, ego_position, copy.deepcopy(gt_image), color_id)
#                 if reshape == True:
#                     reshape_objects(copy.deepcopy(oracle_image), file.split('_')[0], target_path, color, ego_position,copy.deepcopy(gt_image), color_id)
#                 if tlc % 2*time_line_dis ==0:
#                     tlc = 0
#                 if tlc>time_line_dis and tlc<2*time_line_dis :
#                     disap = False
#                     tlc +=1
#                     number_disp = []
#                     ids = []
#                     ids_old = []
#                 else:
#                     disap = True
#                 if disap == True:
#                     ids = disappear_objects(source_path,copy.deepcopy(oracle_image), file.split('_')[0], target_path, color, color_id, ego_position, ids, copy.deepcopy(gt_image))
#                     #Update ids and number of disappearing objects and counting the ids
#                     if len(ids_old)==0:
#                         ids_old = ids
#                         for i in range(len(ids_old)):
#                             number_disp.append(1)
#                     else:
#                         for k, id in enumerate(ids):
#                             if id in ids_old:
#                                 idx = ids_old.index(id)
#                                 number_disp[idx]+=1
#                                 if number_disp[idx]>time_line_dis:
#                                     number_disp.pop(idx)
#                                     ids.remove(id)
#                                     ids_old.remove(id)
#                             else:
#                                 ids_old.append(id)
#                                 number_disp.append(1)
#                         for k, id in enumerate(ids_old):
                            
#                             if id not in ids:
#                                 idx = ids_old.index(id)
#                                 number_disp.pop(idx)
#                                 ids_old.pop(idx)
#                     ids = ids_old
#                     tlc +=1
                            

  