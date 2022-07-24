import argparse
import random
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import glob
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-video", "--video_input", required=True,
	help="path to the input video input")
ap.add_argument("-iquery", "--image_query", required=True,
	help="path to the input image query")
args = vars(ap.parse_args())

# load video input and image query
capture = cv2.VideoCapture(args["video_input"]) # video input
image_query = cv2.imread(args["image_query"]) # query image


# config
MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 1
query = cv2.cvtColor(image_query, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(query, None)
print('[INFO] Number keypoint of image query: ', len(kp1))

def matching(frame):
    kp2, des2 = sift.detectAndCompute(frame, None)
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    if (len(kp2)>MIN_MATCH_COUNT):
        matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = query.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    return frame, len(good)

def check_fps(videocapture):
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver)  < 3 :
        fps = videocapture.get(cv2.cv.CV_CAP_PROP_FPS)
        print("[INFO] Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = videocapture.get(cv2.CAP_PROP_FPS)
        print("[INFO] Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

frame_count = 0
scores = []
start = time.time() # define start time
print('[INFO] Start load video input and process...')
while(capture.isOpened()):
    ret, frame = capture.read()
    if ret == False:
        break
    frame_count += 1
    image, score = matching(frame)
    scores.append(score)
    name_path = 'data/image{0}.jpg'.format(frame_count)
    # store iamge each frame
    cv2.imwrite(name_path, image)

check_fps(capture)
end = time.time() # define time end
print('[INFO] Video input have {0} frame'.format(frame_count))
print('[INFO] Process detect: {:.4f} seconds'.format(end - start))
print('[INFO] Start make video output...')

def make_video(outvid, images=None, fps=30, size=None, is_color=True, format="FMP4"):
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        # if not os.path.exists(image):
        #     raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid

images = list(glob.iglob(os.path.join('data', '*.*')))
# print(len(images))
images = sorted(images, key=lambda x: float((os.path.split(x)[1]).replace('image', '').replace('.jpg', '')))
outvid = os.path.join('data/video', "outvideo.mp4")
make_video(outvid, images, fps=30)

print('[INFO] Finished make video output!!!')

def plot_matching_good(scores):
    if (frame_count != len(scores)):
        print('[ERROR] Number frame in video not correct len of array scores')
    else:
        x = range(frame_count)
        y = scores
        plt.plot(x, y)
        plt.title('Number matching each frame in video')
        plt.xlabel('frame')
        plt.ylabel('matching')
        plt.show()

plot_matching_good(scores)

