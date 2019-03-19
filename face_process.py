import cvlib as cv
import cv2
import pafy
from tqdm import tqdm, trange
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

# for utterly stupid json serialization not working for int32 numpy datatype.. goddam python!
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def detect_faces(image, net, threshold=0.5):
    
    if image is None:
        return None

    (h, w) = image.shape[:2]

    # preprocessing input image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
    net.setInput(blob)

    # apply face detection
    detections = net.forward()

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

def process_video_for_face_landmarks_dlib(filepath, processedfiledir, alphaonly):
    # can't work on a file that doesn't exist!
    if(os.path.isfile(filepath) == False):
        print("File {} does not exist, exiting process_video_for_face_landmarks_dlib...\n".format(filepath))
        return
    
    print("Processing {} for face landmarks".format(filepath))

    p = Path(filepath)
    ext = p.suffix    
   
    writefilename = processedfiledir + "/" + p.stem + "_face_landmarks_temp" + ext
    compressedfilename = processedfiledir + "/" + p.stem + "_face_landmarks_final" + ext

    if(os.path.isfile(writefilename) == True):
        if(os.path.isfile(compressedfilename) == False):
            reencode_video(writefilename,compressedfilename)
        print("File {} already exists, exiting process_video_for_face_landmarks_dlib..\n".format(writefilename))
        return

    # load dlib models etc
    detector = dlib.get_frontal_face_detector()
    cnndetector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    use_cnn_face_detector = True
    use_frontal_face_detector = True

    # open the file    
    capture = cv2.VideoCapture(filepath)
    #capture.open(best.url)

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    # Define the codec and create VideoWriter object.The output is stored in file.


    # use *'MJPG' for the most reliable, but filesize will be insane
    out = cv2.VideoWriter(writefilename,cv2.VideoWriter_fourcc(*'H264'), fps, (frame_width,frame_height))
    ##out = cv2.VideoWriter(writefilename,-1, fps, (frame_width,frame_height))

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    success,image = capture.read()
    currentframe = 0

    progress_bar = tqdm(total=total_frames)

    while success:
        #print( "frame {0} of {1}".format(currentframe,total_frames))
        progress_bar.update()
        currentframe += 1

        frame = imutils.resize(image, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # optionally use DLIB's frontal face detector (HOG+SVG?) to detect faces.. faster than CNN!
        # note: We can use both right now, because I want to see the results
        if use_frontal_face_detector == True:
            # apply face detection using HOG method.. returns a list of rects..
            faces_dlib = detector(gray,0)

            # loop through detected faces
            for face in faces_dlib:
                shape = predictor(gray,face)
                shape = face_utils.shape_to_np(shape)

                for(x,y) in shape:
                    scaledx = int(x * (frame_width/600))
                    scaledy = int(y * (frame_width/600))
                    cv2.circle(image,(scaledx,scaledy),3,(0,255,255),-1)
        
        # optionally do a pass using the CNN based face detector
        if use_cnn_face_detector == True:
            # uses a Maximum-Margin Object Detector (MMOD)
            faces_cnn = cnndetector(gray,0)

            for face in faces_cnn:
                shape = predictor(gray,face.rect)
                shape = face_utils.shape_to_np(shape)

                for(x,y) in shape:
                    scaledx = int(x * (frame_width/600))
                    scaledy = int(y * (frame_width/600))
                    cv2.circle(image,(scaledx,scaledy),3,(255,255,0),-1)
            
        out.write(image)

        #cv2.imshow('frame', image)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        if cv2.waitKey(1) == 27:        
            break

        success,image = capture.read()

    progress_bar.close()
    cv2.destroyAllWindows()
    capture.release()
    out.release()

    reencode_video(writefilename,compressedfilename)

    return compressedfilename


def process_video_for_face_and_bounds(filepath, processedfiledir, alphaonly):
    # can't work on a file that doesn't exist!
    if(os.path.isfile(filepath) == False):
        print("File {} does not exist, exiting process_video_for_face_and_bounds...\n".format(filepath))
        return
    
    print("Processing {} for face and bounds".format(filepath))

    p = Path(filepath)
    ext = p.suffix    
   
    writefilename = processedfiledir + "/" + p.stem + "_temp" + ext
    compressedfilename = processedfiledir + "/" + p.stem + "_final" + ext

    if(os.path.isfile(writefilename) == True):
        if(os.path.isfile(compressedfilename) == False):
            reencode_video(writefilename,compressedfilename)
        print("File {} already exists, exiting process_video_for_face_and_bounds...\n".format(writefilename))
        return
    # access resource files inside package
    prototxt = resource_filename(Requirement.parse('cvlib'),
                                                'cvlib' + os.path.sep + 'data' + os.path.sep + 'deploy.prototxt')
    caffemodel = resource_filename(Requirement.parse('cvlib'),
                                            'cvlib' + os.path.sep + 'data' + os.path.sep + 'res10_300x300_ssd_iter_140000.caffemodel')

    # read pre-trained wieights
    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)


    # open the file    
    capture = cv2.VideoCapture(filepath)
    #capture.open(best.url)

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    # Define the codec and create VideoWriter object.The output is stored in file.


    # use *'MJPG' for the most reliable, but filesize will be insane
    out = cv2.VideoWriter(writefilename,cv2.VideoWriter_fourcc(*'H264'), fps, (frame_width,frame_height))
    ##out = cv2.VideoWriter(writefilename,-1, fps, (frame_width,frame_height))

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    success,image = capture.read()
    currentframe = 0

    progress_bar = tqdm(total=total_frames)

    while success:
        #print( "frame {0} of {1}".format(currentframe,total_frames))
        progress_bar.update()
        currentframe += 1

        # apply face detection
        faces, confidences = detect_faces(image,net,threshold=0.3)

        if(alphaonly == True):
            height, width = image.shape[:2]
            cv2.rectangle(image, (0,0) ,(width,height), (0,0,0), cv2.FILLED)
        #print("face : {}".format(faces))
        #print("confidence : {}".format(confidences))

        # loop through detected faces
        for face,conf in zip(faces,confidences):
            (startX,startY) = face[0],face[1]
            (endX,endY) = face[2],face[3]
            # draw rectangle over face
            cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 4)
            text = "confidence: {0:4.3f}".format(conf)
            cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 

        bbox, label, conf = cv.detect_common_objects(image)
        image = draw_bbox(image, bbox, label, conf)

        out.write(image)

        #cv2.imshow('frame', image)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        if cv2.waitKey(1) == 27:        
            break

        success,image = capture.read()

    progress_bar.close()
    cv2.destroyAllWindows()
    capture.release()
    out.release()

    reencode_video(writefilename,compressedfilename)

    return compressedfilename

def process_video_for_face_and_bounds_to_json(filepath, processedfiledir):
    # can't work on a file that doesn't exist!
    if(os.path.isfile(filepath) == False):
        print("File {} does not exist, exiting process_video_for_face_and_bounds_to_json...\n".format(filepath))
        return

    p = Path(filepath)
    ext = p.suffix    
   
    writefilename = processedfiledir + "/" + p.stem + ".json"

    print("Processing {} for face and bounds and creating json file {} ".format(filepath,writefilename))

    if(os.path.isfile(writefilename) == True):
        print("File {} already exists, exiting process_video_for_face_and_bounds...\n".format(writefilename))
        return writefilename
    # access resource files inside package
    prototxt = resource_filename(Requirement.parse('cvlib'),
                                                'cvlib' + os.path.sep + 'data' + os.path.sep + 'deploy.prototxt')
    caffemodel = resource_filename(Requirement.parse('cvlib'),
                                            'cvlib' + os.path.sep + 'data' + os.path.sep + 'res10_300x300_ssd_iter_140000.caffemodel')

    # read pre-trained wieights
    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    # open the file    
    capture = cv2.VideoCapture(filepath)
    #capture.open(best.url)

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    success,image = capture.read()
    currentframe = 0

    progress_bar = tqdm(total=total_frames)

    jsondata = {}
    jsondata["numframes"] = total_frames
    jsondata["frames"] = {}


    while success:
        #print( "frame {0} of {1}".format(currentframe,total_frames))
        jsonperframedata = {}

        progress_bar.update()

        # apply face detection
        faces, confidences = detect_faces(image,net,threshold=0.3)
        jsonperframedata["faces"] = faces
        jsonperframedata["faceconfidences"] = confidences

        bbox, label, conf = cv.detect_common_objects(image)
        jsonperframedata["bboxes"] = bbox
        jsonperframedata["labels"] = label
        jsonperframedata["confidences"] = conf

        jsondata["frames"][currentframe] = jsonperframedata
        
        currentframe += 1

        #cv2.imshow('frame', image)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        if cv2.waitKey(1) == 27:        
            break

        success,image = capture.read()

    progress_bar.close()
    cv2.destroyAllWindows()
    capture.release()
  
    with open(writefilename, 'w') as f:
        json.dump(jsondata,f,cls=MyEncoder)
        f.close()

    return writefilename



