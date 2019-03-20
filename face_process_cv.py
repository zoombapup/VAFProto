import cvlib as cv
import cv2
import pafy
from cvlib.object_detection import draw_bbox
import string
import numpy as np
import os
from moviepy.editor import *
from pkg_resources import resource_filename, Requirement
from pathlib import Path
import json
import dlib
import imutils
from imutils import face_utils

class FaceDetector_OpenCV_Haar:
    def __init__(self):
        # opencv models
        self.opencv_haar_detector = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
        # shape predictor is dlib's one
        self.dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def DetectFaces(self, image, DrawFaces=True):
        frame_width = image.shape[1]
        frame_height = image.shape[0]
        scale_width = frame_width / 300
        scale_height = frame_height / 300

        # convert the incoming image frame into a grayscale image
        frameOpenCVHaarSmall = cv2.resize(image, (300,300))
        gray = cv2.cvtColor(frameOpenCVHaarSmall, cv2.COLOR_BGR2GRAY)
        # apply face detection using HOG method.. returns a list of rects..
        self.haar_faces = self.opencv_haar_detector.detectMultiScale(gray)
        # loop through detected faces
        for (x,y,w,h) in self.haar_faces:
            x1 = x
            y1 = y
            x2 = x+w
            y2 = y+h
            face = [x,y,w,h]
            cvRect = [int(x1 * scale_width),int(y1*scale_height),int(x2*scale_width),int(y2*scale_height)]
            cv2.rectangle(image, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                      int(round(frame_height / 150)), 4)
            # shape = self.dlib_predictor(gray,face)
            # shape = face_utils.shape_to_np(shape)

            # for(x,y) in shape:
            #     scaledx = int(x * (frame_width/600))
            #     scaledy = int(y * (frame_width/600))
            #     cv2.circle(image,(scaledx,scaledy),3,(0,255,0),-1)



class FaceDetector_OpenCV_SSD:
    def __init__(self):
        prototxt = resource_filename(Requirement.parse('cvlib'),
                                                    'cvlib' + os.path.sep + 'data' + os.path.sep + 'deploy.prototxt')
        caffemodel = resource_filename(Requirement.parse('cvlib'),
                                                'cvlib' + os.path.sep + 'data' + os.path.sep + 'res10_300x300_ssd_iter_140000.caffemodel')
        # read pre-trained wieights
        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    def detect_faces(self, image, net, threshold=0.5):
    
        if image is None:
            return None

        (h, w) = image.shape[:2]

        # preprocessing input image
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
        self.net.setInput(blob)

        # apply face detection
        detections = self.net.forward()

        faces = []
        confidences = []

        # loop through detected faces
        for i in range(0, detections.shape[2]):
            conf = detections[0,0,i,2]

            # ignore detections with low confidence
            if conf < threshold:
                continue

            # get corner points of face rectangle
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype('int')

            faces.append([startX, startY, endX, endY])
            confidences.append(conf)

        # return all detected faces and
        # corresponding confidences    
        return faces, confidences

    def DetectFaces(self,image):
        faces, confidences = self.detect_faces(image,net,0.4)
        # loop through detected faces
        for face,conf in zip(faces,confidences):
            (startX,startY) = face[0],face[1]
            (endX,endY) = face[2],face[3]
            # draw rectangle over face
            cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 4)
            text = "confidence: {0:4.3f}".format(conf)
            cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
            # now lets get the predicted landmarks and draw them
            scaled_face = face
            scaled_face[0] = scaled_face[0] 
            shape = dlib_predictor(gray,face)
            shape = face_utils.shape_to_np(shape)

            for(x,y) in shape:
                scaledx = int(x * (frame_width/600))
                scaledy = int(y * (frame_width/600))
                cv2.circle(image,(scaledx,scaledy),3,(255,0,255),-1)
