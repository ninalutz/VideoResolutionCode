import cv2
from cv2 import dnn_superres
import numpy as np
import os
from moviepy.editor import *

from videowrite import frame_cap, convert_frames_to_video

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read the desired model
amount_scale = 4
model = "FSRCNN_x"
model_src = "fsrcnn"
path = model + str(amount_scale) + ".pb"
sr.readModel(path)

blurs = ["avg", "gaus", "bilat", "med"]
mode = blurs[0]
median_amount = 5
gaus_amount = 5
current_number = 1

gaus_vals = [3, 5, 9, 13, 21]
med_vals = [7, 11, 13]


"""
Applies gaussian blur to a directory
Writes frames to a directory
"""
def gaussian_blur(pathIn):
    files = [f for f in os.listdir(pathIn)]
    for f in files:
        if f[-1] != "g":
            files.remove(f)
    for i in range(len(files)):
        filename=pathIn + str(i) + ".jpg"
        #reading each files
        img = cv2.imread(filename)
        result = cv2.GaussianBlur(img,(gaus_amount,gaus_amount),0)
        cv2.imwrite("gaussianframes/" + str(current_number) +"/" + str(i) + ".jpg", result)
    convert_frames_to_video("gaussianframes/" + str(current_number)+ "/" , "Variations_5_14_21/gaus_"+ str(gaus_amount) + "/" + str(current_number) + "_result_gaussian"+str(gaus_amount)+".mov")

"""
Applies median blur to a directory of frames
Writes frames to a directory
"""
def median_blur(pathIn):
    files = [f for f in os.listdir(pathIn)]
    for f in files:
        if f[-1] != "g":
            files.remove(f)
    for i in range(len(files)):
        filename=pathIn + str(i) + ".jpg"
        #reading each files
        img = cv2.imread(filename)
        result =cv2.medianBlur(img,median_amount)
        cv2.imwrite("medianframes/"  + str(current_number)+ "/"+ str(i) + ".jpg", result)
    convert_frames_to_video("medianframes/"  + str(current_number) +"/", "Variations_5_14_21/med_"+ str(median_amount) + "/" + str(current_number) + "_result_median"+str(median_amount)+".mov")


def test_local():
    for i in range(1, 11):
        current_number = i
        try:
            frame_cap("ResolutionTests/May-Demo-" + str(i) + ".mov", i)
        except Exception as e:
            print(e)
            print("DONE")

        for j in gaus_vals:
            gaus_amount = j
            gaussian_blur("rawframes/"+str(i)+"/")
        for j in med_vals:
            median_amount = j
            median_blur("rawframes/"+str(i)+"/")
