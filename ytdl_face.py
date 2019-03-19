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

def format_filename(s):
    """Take a string and return a valid filename constructed from the string.
Uses a whitelist approach: any characters not present in valid_chars are
removed. Also spaces are replaced with underscores.
 
Note: this method may produce invalid filenames such as ``, `.` or `..`
When I use this method I prepend a date string like '2009_01_15_19_46_32_'
and append a file extension like '.txt', so I avoid the potential of using
an invalid filename.
 
"""
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ','_') # I don't like spaces in filenames.
    return filename

"""
Re-encode video file into a compressed version using ffmpeg params to cut down the file size.. otherwise the video files are huuuge!
"""
def reencode_video(writefilename, compressedfilename):
    # read and write the temp file into a better compressed version.. lame!
    clip = VideoFileClip(writefilename)
    print("Writing compressed file: {}".format(compressedfilename))
    clip.write_videofile(compressedfilename,preset='veryslow',threads=4)
    #clip.write_videofile(compressedfilename,bitrate='200k',preset='veryslow',threads=4)
    #clip.write_videofile(compressedfilename,None,'libx264',False,0,preset='medium',bitrate=15000,threads=4,progress_bar=True)

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

url = "https://www.youtube.com/watch?v=4PsS-8YFSrY"
#"https://www.youtube.com/watch?v=3aDcLu1gC8w"
# "https://www.youtube.com/watch?v=9YQ3nmgO06Q"
# "https://www.youtube.com/watch?v=Src0l4a51sk"
# "https://www.youtube.com/watch?v=G7RgN9ijwE4"
# "https://www.youtube.com/watch?v=-6HYMpmxdaA"
# "https://www.youtube.com/watch?v=OaS-f8hfDD4"
# "https://www.youtube.com/watch?v=4rT5fYMfEUc"
# "https://www.youtube.com/watch?v=HCjPBpdlccM"
# "https://www.youtube.com/watch?v=VYOjWnS4cMY"
# "https://www.youtube.com/watch?v=us1gRCZhShI" 
# "https://www.youtube.com/watch?v=NKpuX_yzdYs" 
# "https://www.youtube.com/watch?v=WCUNPb-5EYI" 
# "https://www.youtube.com/watch?v=NKpuX_yzdYs"
video = pafy.new(url)

best = video.getbestvideo(preftype="mp4")
print(best)
filepath = video.title + "."  + best.extension
filepath = format_filename(filepath)
# download the file if we haven't cache'd it locally already
if(os.path.isfile(filepath) == False):
    print("cache file for {} does not exist, downloading...\n".format(filepath))
    best.download(filepath)


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
path, ext = os.path.splitext(filepath)
writefilename = path + "_temp" + ext
compressedfilename = path + "_final" + ext

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

    #print("face : {}".format(faces))
    #print("confidence : {}".format(confidences))

    # loop through detected faces
    for face,conf in zip(faces,confidences):
        (startX,startY) = face[0],face[1]
        (endX,endY) = face[2],face[3]
        # draw rectangle over face
        cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)
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

cv2.destroyAllWindows()
capture.release()
out.release()

reencode_video(writefilename,compressedfilename)
