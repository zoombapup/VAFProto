import cvlib as cv
import cv2
import pafy
from tqdm import tqdm, trange
from cvlib.object_detection import draw_bbox
import string
import numpy as np
import json
import os
from moviepy.editor import *
from pkg_resources import resource_filename, Requirement
from ytdl_pafy_process import *
from pathlib import Path
import subprocess
import shutil

def process_video_for_openpose(filepath, processedfiledir, disable_blending):
    # can't work on a file that doesn't exist!
    if(os.path.isfile(filepath) == False):
        print("File {} does not exist, exiting process_video_for_openpose...\n".format(filepath))
        return
    
    print("Processing {} for open pose".format(filepath))

    p = Path(filepath)
    ext = p.suffix    
   
    writefilename = processedfiledir + "/" + p.stem + "_pose_temp" + ext
    compressedfilename = processedfiledir + "/" + p.stem + "_pose_final" + ext

    if(os.path.isfile(writefilename) == True):
        if(os.path.isfile(compressedfilename) == False):
            reencode_video(writefilename,compressedfilename)
        print("File {} already exists, exiting process_video_for_openpose...\n".format(writefilename))
        return
    
    openpose_exe = "OpenPoseDemo.exe"
    path_to_openpose_exe = "F:\\Phils Research Codebases\\OpenPose\\PhilsOpenPose\\Build\\bin\\"

    absolute_filepath = os.path.abspath(filepath)
    absolute_writefilename = os.path.abspath(writefilename)

    print("Processing {} with openpose and writing {}".format(absolute_filepath,absolute_writefilename))
    # call OpenPose as external application
    subprocess.run(args=[path_to_openpose_exe + openpose_exe, "--video", absolute_filepath, "--write_video", absolute_writefilename, "--display","0", "--disable_blending", disable_blending],cwd=path_to_openpose_exe)

    reencode_video(writefilename,compressedfilename)

    return compressedfilename


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

# Writes openpose data to json file using itermediate json files+dir written by openpose C++ app, then deletes intermediate
# directory after the json files have been consolidated.

def process_video_for_openpose_to_json(videofilename,processeddir, jsonfilename,disable_blending):
    # can't work on a file that doesn't exist!
    if(os.path.isfile(videofilename) == False):
        print("File {} does not exist, exiting process_video_for_openpose_to_json...\n".format(videofilename))
        return
    
    print("Processing {} for open pose and writing to json file {} ".format(videofilename,jsonfilename))

    p = Path(jsonfilename)
    ext = p.suffix    
   
    jsonfolder = processeddir + "/" + p.stem + "/"

    openpose_exe = "OpenPoseDemo.exe"
    path_to_openpose_exe = "F:\\Phils Research Codebases\\OpenPose\\PhilsOpenPose\\Build\\bin\\"

    absolute_filepath = os.path.abspath(videofilename)
    absolute_jsonfoldername = os.path.abspath(jsonfolder)

    print("Processing {} with openpose and writing {}".format(absolute_filepath,absolute_jsonfoldername))
    # call OpenPose as external application
    subprocess.run(args=[path_to_openpose_exe + openpose_exe, "--video", absolute_filepath, "--write_json", absolute_jsonfoldername, "--display","0", "--render_pose", "0", "--disable_blending", disable_blending, "--face","--hand"],cwd=path_to_openpose_exe)

    # the above call will have generated a folder with json files per frame in.. now we consolidate those files back into the input json file
    # and then remove the folder so we don't have crap everywhere

    main_json_data = ParseJsonFile(jsonfilename)
    # in case the main json file doesn't exist yet (this is first method to write to it for instance)
    if main_json_data is None:
        main_json_data = {}

    if "PoseFrames" not in main_json_data:
        main_json_data["PoseFrames"] = []
    else:
        main_json_data["PoseFrames"].clear()

    for file in os.listdir(absolute_jsonfoldername):
        if file.endswith(".json"):
            jsonfilepath = os.path.join(absolute_jsonfoldername, file)
            p = Path(jsonfilepath)
            print("Processing file: {} with path: {} ".format(file,jsonfilepath))
            json_data = ParseJsonFile(jsonfilepath)
            main_json_data["PoseFrames"].append(json_data)

    # delete the temp directory openpose wrote to
    shutil.rmtree(absolute_jsonfoldername,ignore_errors=True)

    # write back the json file
    WriteJsonFile(jsonfilename,main_json_data)

    return jsonfilename


