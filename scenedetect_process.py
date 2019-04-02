from tqdm import tqdm, trange
import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors.content_detector import ContentDetector
from moviepy.editor import *
import string
import numpy as np
import os
from pkg_resources import resource_filename, Requirement
from pathlib import Path
import json
from json_utils import *

import moviepy.config as mpconf

mpconf.change_settings()

# Process video using pyscenedetect and save the scenes to json file for later use
def process_video_for_scene_detection(filepath, jsonfilename):
    # can't work on a file that doesn't exist!
    if(os.path.isfile(filepath) == False):
        print("File {} does not exist, exiting process_video_for_scene_detection...\n".format(filepath))
        return
    
    print("Processing {} and detecting scenes".format(filepath))

    main_json_data = ParseJsonFile(jsonfilename)
    # in case the main json file doesn't exist yet (this is first method to write to it for instance)
    if main_json_data is None:
        main_json_data = {}
    
    if "Scenes" not in main_json_data:
        main_json_data["Scenes"] = []
    else:
        main_json_data["Scenes"].clear()

    # videomanager takes a Python list of filenames, so we create a list and add our one file name to that
    absolute_filepath = os.path.abspath(filepath)
    filelist = list()
    filelist.append(absolute_filepath)
    # do the scene detection thang
    video_manager = VideoManager(filelist)
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())
    start_timecode = video_manager.get_base_timecode()

    try:
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        cut_list = scene_manager.get_cut_list(start_timecode)
        print('List of cuts obtained:')
        for i, cut in enumerate(cut_list):
            print('    Cut %2d: Time: %s / Frame: %d' % ( i+1,cut.get_timecode(), cut.get_frames() ) )

        scene_list = scene_manager.get_scene_list(start_timecode)
        print('List of scenes obtained:')
        for i, scene in enumerate(scene_list):
            print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
                i+1,
                scene[0].get_timecode(), scene[0].get_frames(),
                scene[1].get_timecode(), scene[1].get_frames(),))
            scene_dict = {}
            scene_dict["start_timecode"] = scene[0].get_timecode()
            scene_dict["end_timecode"] = scene[1].get_timecode()
            scene_dict["start_frame"] = scene[0].get_frames()
            scene_dict["end_frame"] = scene[1].get_frames()
            main_json_data["Scenes"].append(scene_dict)

    finally:
        video_manager.release()

    # write back the json file
    WriteJsonFile(jsonfilename,main_json_data)

def render_scenes_to_video(filepath,processedfiledir, jsonfilename):

      # can't work on a file that doesn't exist!
    if(os.path.isfile(filepath) == False):
        print("File {} does not exist, exiting render_scenes_to_video...\n".format(filepath))
        return
    
    print("Processing {} and writing temp video for detected scenes".format(filepath))

    main_json_data = ParseJsonFile(jsonfilename)
    # in case the main json file doesn't exist yet (this is first method to write to it for instance)
    if main_json_data is None:
        main_json_data = {}
    
    if "Scenes" in main_json_data:
        # we have scene info.. process the input video in a loop and write on the scene number at top left
        currentframeindex = 0

        p = Path(filepath)
        ext = p.suffix    
   
        writefilename = processedfiledir + "/" + p.stem + "_scenes_temp" + ext

        mainclip = VideoFileClip(filepath)
        text = []
        for scene in main_json_data["Scenes"]:
            # create a new moviepy image clip with the scene number on it
            text.append(TextClip(str(currentframeindex),fontsize=40,color='white',method='label',).set_start(scene["start_timecode"],False).set_end(scene["end_timecode"]))
            currentframeindex = currentframeindex + 1

        textclips = concatenate_videoclips(text)
        compositeclip = CompositeVideoClip([mainclip,textclips])

        compositeclip.write_videofile(writefilename)





