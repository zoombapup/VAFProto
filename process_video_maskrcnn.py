from process_video_template import *
import cv2
import numpy as np
import process_video_template
# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
import sys, os.path
maskrcnnpath = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/Mask_RCNN/')
sys.path.append(".")
sys.path.append(maskrcnnpath)
from visualize_cv2 import model, display_instances, class_names

class MaskRCNN_VideoProcessor(VideoProcessor):
    def __init__(self):
        VideoProcessor.__init__(self)
        # do stuff to initialize here
        # this is the name that will be used to identify the video processor in console output
        self.processorname = "MaskRCNNVideoProcessor"
        self.per_frame_results = []

    def Process_Frame(self, image, frame_count):
        # add mask to frame
        results = model.detect([image], verbose=0)
        r = results[0]
        image = display_instances(
            image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        self.Write_Output_Frame(image, frame_count)
        self.per_frame_results.append(frame_count)
        return image