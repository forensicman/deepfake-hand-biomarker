import mediapipe as mp
import numpy as np
import math
import cv2
from os import listdir, remove
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

corepath = 'C:/Users/user/Projects/project/'
img_path_real_hands = corepath + 'hands_real/'
img_path_real_hands_crop = corepath + 'hands_real_crop/'
img_path_fake_hands = corepath + 'hands_fake/'
img_path_fake_hands_crop = corepath + 'hands_fake_crop/'



def scale_img(img_convert, hand_landmarks=[]):
    height, width, _ = img_convert.shape
    #Get all x and y coordinates form hand landmarks into array
    x_coord = [landmark.x for landmark in hand_landmarks]
    y_coord = [landmark.y for landmark in hand_landmarks]
    #Find the min and max of x and y and scale with width and height to revert normalization
    start_x = int(min(x_coord) * width)
    start_y = int(min(y_coord) * height)
    end_x = int(max(x_coord) * width)
    end_y = int(max(y_coord) * height)
    #Get the distance of start and end pixel. This is needed calculate the margin
    start_end_dist = math.dist((start_x, start_y), (end_x, end_y))
    #Based on the distance copped image will be enlarged by 10 percent
    margin = int(10 * start_end_dist / 100)
    start_x = start_x - margin
    start_y = start_y - margin
    end_x = end_x + margin
    end_y = end_y + margin

    #Make sure x and y are not below 0 and not exceed width and height
    if start_x < 0:
        start_x = 0
    if start_x > width:
        start_x = width
    if start_y < 0:
        start_y = 0
    if start_y > height:
        start_y = height

    crop_img = img_convert[start_y:end_y, start_x:end_x]

    return crop_img


def hand_landmarker_detect(img_path, img_path_crop):

    base_options = python.BaseOptions(model_asset_path='models/gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=4)
    recognizer = vision.GestureRecognizer.create_from_options(options)
    min_shape_crop = (35,35,3)

    for image in listdir(img_path):
        img = img_path + image
        try:
            img_in = mp.Image.create_from_file(img)
        except:
            print("Error Image create for detection: " +img)
        
        detection_result = recognizer.recognize(img_in)

        if len(detection_result.hand_landmarks) == 0:
            remove(img)
        
        for i in range(len(detection_result.hand_landmarks)):
            hand_landmarks = detection_result.hand_landmarks[i]

            try:
                img_convert = cv2.imread(img)
            except:
                print("Error reading: " + img)

            gesture = detection_result.gestures[0][0].category_name

            if not gesture:
                continue

            crop_img = scale_img(img_convert, hand_landmarks)
            crop_img_shape = crop_img.shape

            if crop_img_shape < min_shape_crop:
                print("too small")
                continue
            file_path = Path(img)
            file_extension = file_path.suffix
            img_hand_split = image.split(file_extension)[0].replace('.', '') + '-' + str(i) + '-' + str(gesture) + "-" + str(crop_img_shape[0]) + 'x' + \
                             str(crop_img_shape[1]) + file_extension

            try:
                cv2.imwrite(img_path_crop + img_hand_split, crop_img)
            except:
                print("Write error: " + img_path_crop + img_hand_split)

    return

def average_shape(img_path):
    dim1 = []
    dim2 = []
    for image in listdir(img_path):
        img = cv2.imread(img_path + image)
        d1, d2, colors = img.shape
        dim1.append(d1)
        dim2.append(d2)

    print(str(np.mean(dim1)) + " x " + str(np.mean(dim2)))

    return

#Examples to call the functions below

#hand_landmarker_detect(img_path_real_hands, img_path_real_hands_crop)
#hand_landmarker_detect(img_path_fake_hands, img_path_fake_hands_crop)

#average_shape(img_path_real_hands_crop)
#average_shape(img_path_fake_hands_crop)