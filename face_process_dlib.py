import cvlib as cv
import cv2
import pafy
from cvlib.object_detection import draw_bbox
import string
import numpy as np
import os
from moviepy.editor import *
from pkg_resources import resource_filename, Requirement
from ytdl_pafy_process import *
from pathlib import Path
import json
import dlib
import imutils
from imutils import face_utils

class FaceDetector_DLib_Hog:
    def __init__(self):
        self.dlib_hog_detector = dlib.get_frontal_face_detector()
        # shape predictor is dlib's one
        self.dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def DetectFaces(self, image, DrawFaces=True):
        frame_width = image.shape[1]
        # convert the incoming image frame into a grayscale image
        frame = imutils.resize(image, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # apply face detection using HOG method.. returns a list of rects..
        faces_dlib = self.dlib_hog_detector(gray,0)

        # loop through detected faces
        for face in faces_dlib:
            shape = self.dlib_predictor(gray,face)
            shape = face_utils.shape_to_np(shape)

            for(x,y) in shape:
                scaledx = int(x * (frame_width/600))
                scaledy = int(y * (frame_width/600))
                cv2.circle(image,(scaledx,scaledy),3,(0,255,255),-1)

class FaceDetector_DLib_CNN:
    def __init__(self):
            # load dlib models etc
            self.dlib_cnn_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
            # shape predictor is dlib's one
            self.dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def DetectFaces(self, image, DrawFaces=True):
        frame_width = image.shape[1]
        # convert the incoming image frame into a grayscale image
        frame = imutils.resize(image, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # uses a Maximum-Margin Object Detector (MMOD)
        faces_cnn = self.dlib_cnn_detector(gray,0)

        for face in faces_cnn:
            shape = self.dlib_predictor(gray,face.rect)
            shape = face_utils.shape_to_np(shape)

            for(x,y) in shape:
                scaledx = int(x * (frame_width/600))
                scaledy = int(y * (frame_width/600))
                cv2.circle(image,(scaledx,scaledy),3,(255,255,0),-1)