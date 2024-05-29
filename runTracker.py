# combination of detectNet and actionNet

#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse
import cv2
import jetson_inference
from jetson_inference import detectNet
import jetson_utils
from jetson_utils import videoSource, videoOutput, Log

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# create video sources and outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)
	
# load the object detection network
net = detectNet(args.network, sys.argv, args.threshold)

shotAttempts = 0 # shot attempt counter
cooldown = 0 # cooldown timer between shots

# detects bounding box intersection between ball and hoop
def intersects(rect1, rect2):
	return not (rect1[2] < rect2[0] or rect1[0] > rect2[2] or rect1[3] < rect2[1] or rect1[1] > rect2[3])


# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  
        
    # detect objects in the image (with overlay)
    detections = net.Detect(img, overlay=args.overlay)
    
    basketballBbox = None
    rimBbox = None
    for detection in detections:
        if detection.ClassID == 1: # basketball
        	basketballBbox = detection
        elif detection.ClassID == 2: # hoop
        	rimBbox = detection
        	
    # check cooldown and intersection
    if basketballBbox and rimBbox and cooldown == 0:
    	basketballRect = (basketballBbox.Left, basketballBbox.Top, basketballBbox.Right, basketballBbox.Bottom)
    	rimRect = (rimBbox.Left, rimBbox.Top, rimBbox.Right, rimBbox.Bottom)
    	
    	if intersects(basketballRect, rimRect):
    		shotAttempts += 1
    		cooldown = 150
    
    if cooldown > 0:
    	cooldown -= 1
    
    # convert image to numpy
    img = jetson_utils.cudaToNumpy(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    # overlay shot counter
    cv2.putText(img, f'Shot Counter: {shotAttempts}', (1000,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) 

    # render the image
    img = jetson_utils.cudaFromNumpy(img)
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
