# Video Analysis Framework Prototype
Video Analysis Framework Prototype Code

This code is poor, but it isn't intended for production, but rather to get used to using python code for computer vision use.

This code will eventually be superceded by a proper video processing pipeline based off a task queue (once I find one that actually works on linux AND windows).

Currently, this code processes a json file (test.json) and uses a whole bunch of different libraries to download video's from Youtube URL's for later analysis. Things like pose detection, facial landmarks, object bounding boxes etc. 

TODO: 
Add back in MaskRCNN based image segmentation
Spit out bounding boxes for faces with confidence values (basically uncomment code and see if it still works)
Fix the openCV usage for video writing, OpenCV writes really huge files (I reencode them to a smaller version using moviepy after finishing the read-process-write loop, but that's not great)
Fix speed of the Facial Landmark detection etc.
Add openface landmark detection
Fix OpenPose so that it spits out some update info during processing
