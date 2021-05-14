import cv2
from cv2 import dnn_superres
import numpy as np
import os
from moviepy.editor import *

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read the desired model
amount_scale = 4
model = "FSRCNN_x"
model_src = "fsrcnn"
path = model + str(amount_scale) + ".pb"
sr.readModel(path)

gaus_amount = 15


# Set the desired model and scale to get correct pre- and post-processing
# sr.setModel(model_src, amount_scale)

# for i in range(1, 11):
#     # Read image
#     test_file = str(i) + ".png"
#     image = cv2.imread('ResolutionTests/' + test_file)

#     # Upscale the image
#     result = sr.upsample(image)

#     # Save the image
#     cv2.imwrite("upscaled_" +test_file + "_" + str(amount_scale), result)
#     print("Done with " + str(i) )

for i in range(1, 11):

    test_file = str(i) + ".png"
    image = cv2.imread('ResolutionTests/' + test_file)

    result = cv2.blur(image,(gaus_amount,gaus_amount))

    cv2.imwrite("average_" + str(gaus_amount) + "_" + test_file, result)
    print("Done with " + str(i) )

# Function to extract frames
def frame_cap(path):
	# Path to video file
	vidObj = cv2.VideoCapture(path)

	# Used as counter variable
	count = 0
  
	# checks whether frames were extracted
	success = 1
  
	while success:
  
		# vidObj object calls read
		# function extract frames
		success, image = vidObj.read()

		cv2.imwrite("frame%d.jpg" % count, image)

		# Upscale the image
		result = sr.upsample(image)

		# Save the image
		cv2.imwrite("frame_up%d.jpg"  % count, result)
  
		#TODO - write out to a video file
  
		count += 1


def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


# frame_cap("16_10_3.mp4")
