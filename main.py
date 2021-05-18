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

gaus_vals = [3, 5, 9, 13, 21]
med_vals = [7, 11, 13]

multi_vals = [3, 5, 7, 9]


"""
Applies gaussian blur to a directory
Writes frames to a directory
"""
def gaussian_blur(pathIn, index):
    files = [f for f in os.listdir(pathIn)]
    for f in files:
        if f[-1] != "g":
            files.remove(f)
    for i in range(len(files)):
        filename=pathIn + str(i) + ".jpg"
        #reading each files
        img = cv2.imread(filename)
        result = cv2.GaussianBlur(img,(gaus_amount,gaus_amount),0)
        cv2.imwrite("gaussianframes/" + str(index) +"/" + str(i) + ".jpg", result)
    convert_frames_to_video("gaussianframes/" + str(index)+ "/" , "Variations_5_14_21/gaus_"+ str(gaus_amount) + "/" + str(index) + "_result_gaussian"+str(gaus_amount)+".mov")

"""
Applies median blur to a directory of frames
Writes frames to a directory
"""
def median_blur(pathIn, index):
    files = [f for f in os.listdir(pathIn)]
    for f in files:
        if f[-1] != "g":
            files.remove(f)
    for i in range(len(files)):
        filename=pathIn + str(i) + ".jpg"
        #reading each files
        img = cv2.imread(filename)
        result =cv2.medianBlur(img,median_amount)
        cv2.imwrite("medianframes/"  + str(index)+ "/"+ str(i) + ".jpg", result)
    convert_frames_to_video("medianframes/"  + str(index) +"/", "Variations_5_14_21/med_"+ str(median_amount) + "/" + str(index) + "_result_median"+str(median_amount)+".mov")


"""
Just sharpens the frames -- no blurring
"""
def sharpen(pathIn, index):
    print(index)
    #Create the identity filter, but with the 1 shifted to the right!
    kernel = np.zeros( (9,9), np.float32)
    kernel[4,4] = 2.0   #Identity, times two! 

    #Create a box filter:
    boxFilter = np.ones( (9,9), np.float32) / 81.0

    kernel = kernel - boxFilter

    files = [f for f in os.listdir(pathIn)]
    for f in files:
        if f[-1] != "g":
            files.remove(f)
    for i in range(len(files)):
        filename=pathIn + str(i) + ".jpg"
        #reading each files
        img = cv2.imread(filename)
        result= cv2.filter2D(img, -1, kernel)
        # result =cv2.medianBlur(custom, 5)
        cv2.imwrite("sharpenframes/"  + str(index)+ "/"+ str(i) + ".jpg", result)
    convert_frames_to_video("sharpenframes/"  + str(index) +"/", "Variations_5_14_21/sharpen/" + str(index) +".mov")


"""
Sharpens and adds blur to images 
"""
def sharpen_and_blur(pathIn, blur_type, blur_amount, index):
    #Create the identity filter, but with the 1 shifted to the right!
    kernel = np.zeros( (9,9), np.float32)
    kernel[4,4] = 2.0   #Identity, times two! 

    #Create a box filter:
    boxFilter = np.ones( (9,9), np.float32) / 81.0

    kernel = kernel - boxFilter

    files = [f for f in os.listdir(pathIn)]
    for f in files:
        if f[-1] != "g":
            files.remove(f)
    for i in range(len(files)):
        filename=pathIn + str(i) + ".jpg"
        #reading each files
        img = cv2.imread(filename)
        sharper = cv2.filter2D(img, -1, kernel)
        if blur_type == "median":
            result =cv2.medianBlur(sharper, blur_amount)
            cv2.imwrite("sharpenframes/"  + str(index)+ "/"+ str(i) + ".jpg", result)
        if blur_type == "gaus":
            result = cv2.GaussianBlur(sharper,(blur_amount,blur_amount),0)
            cv2.imwrite("sharpenframes/"  + str(index)+ "/"+ str(i) + ".jpg", result)
    convert_frames_to_video("sharpenframes/"  + str(index) +"/", "Variations_5_14_21/sharpen_" + blur_type + "_" + str(blur_amount) + "/" + str(index) +".mov")



def test_local():
    for i in range(1, 11):
        try:
            frame_cap("ResolutionTests/May-Demo-" + str(i) + ".mov", i)
        except Exception as e:
            print(e)
            print("DONE")

        for j in multi_vals:
            sharpen_and_blur("rawframes/"+str(i)+"/", "median", j, i)
            sharpen_and_blur("rawframes/"+str(i)+"/", "gaus", j, i)


        # for j in gaus_vals:
        #     gaus_amount = j
        #     gaussian_blur("rawframes/"+str(i)+"/", i)
        # for j in med_vals:
        #     median_amount = j
        #     median_blur("rawframes/"+str(i)+"/", i)

test_local()
