import TIS
import cv2
import numpy as np
import os
import argparse

# Open camera, set video format, framerate and determine, whether the sink is color or bw
# Parameters: Serialnumber, width, height, framerate , color
# If color is False, then monochrome / bw format is in memory. If color is True, then RGB32
# colorformat is in memory
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--trig", help="Enable trigger mode for captures",
                    action="store_true")
parser.add_argument("-w", "--width", help="Width of capture, 256-2048",
                    type=int, default=2048)
parser.add_argument("-e", "--height", help="Height of capture, 4-1536",
                    type=int, default=1536)
parser.add_argument("-f", "--fps", help="Capture frames per second, 1-60, unsupported in trigger mode",
                    type=int, default=15)
args = parser.parse_args()
cameras = []
images = []
if args.trig:
    print("Triggers enabled")

# Camera indexes:
# 0 - Left
# 1 - Right
cameras.append(TIS.TIS("26910053", args.width, args.height, args.fps,1, args.trig)) #Left
cameras.append(TIS.TIS("26910054", args.width, args.height, args.fps,1, args.trig)) #Right

print('Press Esc to stop, press c to capture')
directory = "./captures"
if not os.path.exists(directory):
    os.makedirs(directory)

# Start the pipeline
for cam in cameras:
    cam.Start_pipeline()

cv2.waitKey(500)

# Remove comment below in oder to get a propety list.
# cameras[0].List_Properties()

# Query the gain auto and current value :
print("Gain Auto : %s " % cameras[0].Get_Property("Gain Auto").value)
print("Gain : %d" % cameras[0].Get_Property("Gain").value)
print("Exposure Auto : %d" % cameras[0].Get_Property("Exposure Auto").value)
print("Cameras active, waiting for first frame...")

error = 0
lastkey = 0
img_cntr = 0
capture_pending = False
try:
    while (lastkey != 27):
        if (lastkey == 99):
            capture_pending = True

        all_images_ready = True
        for cam in cameras:
            if not cam.Is_image_ready():
                all_images_ready = False

        if all_images_ready:
            images.clear()
            # Grab an image
            error = 0
            for cam in cameras:
                images.append(cam.Get_image())
 
            if any(elem is None for elem in images):
                print( "ERR: Capture failed")
            else:
                # Do the preview first
                for idx, img in enumerate(images):
                    img_preview = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
                    img_preview = cv2.resize(img_preview, (640,480))
                    if (idx == 0):
                        img_h_concat = img_preview
                    else:
                        img_h_concat = np.concatenate((img_h_concat, img_preview), axis=1)
                cv2.imshow('Stereo Cams', img_h_concat)
                # Save the image if c key is hit
                if (capture_pending):
                    for idx, img in enumerate(images):
                        if (idx == 0):
                            filename = os.path.join(directory, 'img_left_'+ str(img_cntr) +'.npy' )
                        else:
                            filename = os.path.join(directory, 'img_right_'+ str(img_cntr) +'.npy' )
                        #cv2.imwrite('img_left_'+ str(img_cntr) +'.jpg' , img)
                        np.save(filename , img )
                    img_cntr = img_cntr + 1
                    print("Saving image index " + str(img_cntr))
                    print('\a')
                    capture_pending = False

        # wait a few ms for key hit
        lastkey = cv2.waitKey(20) # reduce this value if going above 50fps
                
except KeyboardInterrupt:
        cv2.destroyWindow('Window')

# Stop the pipeline and clean ip
for cam in cameras:
    cam.Stop_pipeline()
print('Program ended')

