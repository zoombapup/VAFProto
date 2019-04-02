from tqdm import tqdm, trange
from moviepy.editor import *
import string
import numpy as np
import os
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer

''' Template for video processor classes.. '''

class VideoProcessor:
    def __init__(self):
        # do stuff to initialize here
        # this is the name that will be used to identify the video processor in console output
        self.processorname = "PassthroughVideoProcessor"
        self.per_frame_results = []

    # open the input video and iterate over its frames, calling Process_Frame on each frame of the video and optionally write results
    # to another video (or the original I guess?)
    def Process_Video(self, inputfilepath, jsonfilename, processedfiledir, outputVideo = False, bitrate=20000 ):
         # can't work on a file that doesn't exist!
        if(os.path.isfile(inputfilepath) == False):
            print("File {} does not exist, exiting {}...\n".format(inputfilepath,self.processorname))
            return

        print("Processing {} using {}".format(inputfilepath, self.processorname))

        # get the input video file and create a temporary filename for the output "temp" file that displays intermediate results
        p = Path(inputfilepath)
        ext = p.suffix    
        writefilename = processedfiledir + "/" + "_" + p.stem + "_" + self.processorname + "_temp" + ext

        # open the input video with moviepy and iterate over the frames of the video
        # process each frame and optionally write out the results to a temporary video for inspection
        # we also update a python array with the results for each frame
        self.video_clip = VideoFileClip(inputfilepath, audio=False)
        self.video_writer = ffmpeg_writer.FFMPEG_VideoWriter(writefilename, self.video_clip.size, self.video_clip.fps, codec="libx264",
                                                        preset="veryfast", bitrate="666k",
                                                        audiofile=inputfilepath, threads=None,
                                                        ffmpeg_params=None)

        frame_count = 0  # The current frame count
        for frame in self.video_clip.iter_frames(progress_bar=True,dtype="uint8"):
            self.Process_Frame(frame, frame_count)
            frame_count += 1

        self.video_writer.close()
        return self.per_frame_results



    def Write_Output_Frame(self, outputimage, frame_count):
        # do the actual processing of the input data.. for a given frame of the video
        self.video_writer.write_frame(outputimage)
        

    def Process_Frame(self, image, frame_count):
        self.Write_Output_Frame(image, frame_count)
        self.per_frame_results.append(frame_count)
        return image