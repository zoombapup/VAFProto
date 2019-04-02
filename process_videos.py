import json
from ytdl_pafy_process import *
from face_process import *
from process_openpose_extern import *
import os
import numpy
from json_utils import *
from scenedetect_process import *
from process_video_template import *
from process_video_maskrcnn import *

## empty python dict - will get filled with data for choosing which videos to process etc.p
data_to_process = {}

        
# json file contains all the files to download and/or process (downloads are cached so we don't redownload things)
data_to_process = ParseJsonFile("test.json")
##if(data_to_process is not None):
##    print(json.dumps(data_to_process, indent=4))

videocachedir = "./cachedvideos"
unprocesseddir = "./unprocessedvideos"
processeddir = "./processedvideos"
final_file_path = ""

for video_info in data_to_process["videos"]:
    if "video_name" in video_info:
        clipname = video_info["video_name"]
        print("Processing video file: {}".format(clipname))
        final_file_path = videocachedir + "/" + clipname
    else:
        print("Processing video file: {}".format(video_info["video_url"]))
        clipname = video_info["video_url"].split("=")[1]
        # download and cache the video at the given url if we havent already
        final_file_path = download_and_cache_video(videocachedir,video_info["video_url"], clipname)

    print("Final file path: {}".format(final_file_path))

    if "subclips" in video_info and len(video_info["subclips"]) > 0:
        print("We have subclips... processing those... ")
        for subclip in video_info["subclips"]:
            subclipname = "{0}-{1:05d}-{2:05d}.mp4".format(clipname,int(convert_to_secs(subclip["start"])),int(convert_to_secs(subclip["end"])))
            print("Processing clip: {0}".format(subclipname))
            create_subclip(final_file_path,unprocesseddir,subclip["start"],subclip["end"],subclipname)
    else:
        print("No subclips, processing whole video")
        copy_video(final_file_path,unprocesseddir,clipname + ".mp4")
        
    # now process the various clips into "temp" copies in the processed folder
    for file in os.listdir(unprocesseddir):
        if file.endswith(".mp4"):
            if clipname in file:
                filepathname = os.path.join(unprocesseddir, file)
                p = Path(filepathname)
                jsonfilename = processeddir + "/" + p.stem + ".json"
                print("Processing video file: {} with json output file: {} ".format(filepathname,jsonfilename))

                results = VideoProcessor().Process_Video(filepathname,jsonfilename,processeddir)
                #print(results)
                maskresults = MaskRCNN_VideoProcessor().Process_Video(filepathname,jsonfilename,processeddir)

                # write scene list to json file
                if "writeScenes" in video_info and video_info['writeScenes'] == "True":
                    process_video_for_scene_detection(filepathname,jsonfilename)
                    render_scenes_to_video(filepathname,processeddir,jsonfilename)



                # now we look at the various processing options from the json data and use them 
                #if "writeFaceLandmarks" in video_info and video_info['writeFaceLandmarks'] == "True":
                if IsJsonValueTrue(video_info,"writeFaceLandmarks"):
                    process_video_for_face_landmarks_dlib(filepathname,processeddir,True)
                #process_video_for_face_and_bounds(filepathname,processeddir,True)   # third param is if we want to reset image frame to black and alpha-only draw CV info
                #process_video_for_openpose(filepathname, processeddir)
                #fname = process_video_for_face_and_bounds_to_json(filepathname,processeddir)
                disable_blending = "0"
                if "disable_blending" in video_info:
                    disable_blending = video_info["disable_blending"]
                        
                if "writePose" in video_info and video_info['writePose'] == "True":
                    if "writePoseJson" in video_info:
                        process_video_for_openpose_to_json(filepathname, processeddir, jsonfilename,disable_blending)
                    else: 
                        print("Ignoring video file: {} for pose ouput to json output file: {} ".format(filepathname,jsonfilename))
                        process_video_for_openpose(filepathname, processeddir,disable_blending)
                else:
                    print("Ignoring video file: {} for pose ouput {} as writePose was not True ".format(filepathname,jsonfilename))
        
