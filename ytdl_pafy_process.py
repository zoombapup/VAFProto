import pafy
from tqdm import tqdm, trange
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
    clip.close()
    #clip.write_videofile(compressedfilename,bitrate='200k',preset='veryslow',threads=4)
    #clip.write_videofile(compressedfilename,None,'libx264',False,0,preset='medium',bitrate=15000,threads=4,progress_bar=True)

"""
Download a video if not already in a cache dir, return the (relative) pathname of the downloaded file
"""

def download_and_cache_video(cachedir, url, filename):
    video = pafy.new(url)
    #best = video.getbestvideo(preftype="mp4")
    best = video.getbest(preftype="mp4")
    print(best)
    filepath = cachedir + "/" + filename + "."  + best.extension
    #filepath = format_filename(filepath)
    # download the file if we haven't cache'd it locally already
    if(os.path.isfile(filepath) == False):
        print("cache file for {} does not exist, downloading...\n".format(filepath))
        best.download(filepath)
    return filepath

"""
Given a video file, create subclip of it in unprocessed directory
"""
def create_subclip(final_file_name,unprocessedfiledir,start,end,subclipfilename):
    if(os.path.isfile(final_file_name) == False):
        print("Can't open video file {} to create subclip {}", final_file_name,subclipfilename)
        return
    filepath = unprocessedfiledir + "/" + subclipfilename
    if(os.path.isfile(filepath) == False):
        clip = VideoFileClip(final_file_name)
        newclip = clip.subclip(start,end)
        name = unprocessedfiledir + "/" + subclipfilename
        newclip.write_videofile(name,preset='slow',threads=4)
        newclip.close()
        clip.close()
    else:
        print("Subclip {} already exists.. ignoring".format(subclipfilename))

"""
Given a video file, copy it to the unprocessed directory if it doesn't already exist
"""
def copy_video(final_file_name, unprocessedfiledir,clipfilename):
    if(os.path.isfile(final_file_name) == False):
        print("Can't open video file {} for copy operation! awooga!", final_file_name)
        return
    filepath = unprocessedfiledir + "/" + clipfilename
    if(os.path.isfile(filepath) == False):
        clip = VideoFileClip(final_file_name)
        clip.write_videofile(filepath,preset='slow',threads=4)
        clip.close()
    else:
        print("Clip {} already exists.. ignoring".format(clipfilename))

def convert_to_secs(time):
    return cvsecs(time)