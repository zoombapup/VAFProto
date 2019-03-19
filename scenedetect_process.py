from tqdm import tqdm, trange
import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors.content_detector import ContentDetector
import string
import numpy as np
import os
from pkg_resources import resource_filename, Requirement
from pathlib import Path
import json

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

def ParseJsonFile(filepath):
    try:
        with open(filepath) as json_file:
            data = json.load(json_file)
        return data    
    except FileNotFoundError:
        print("File Not found while loading Json file {}".format(filepath))


def WriteJsonFile(filepath, json_data):
    try:
        with open(filepath,'w') as json_file:
            json.dump(json_data,json_file)
    except IOError:
        print("IO Error Writing Json file {}".format(filepath))

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
  