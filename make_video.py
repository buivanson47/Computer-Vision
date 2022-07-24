import cv2
import numpy as np
import os
import glob
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize

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
print(len(images))
# x = images[0]
# print((os.path.split(x)[1]).replace('image', '').replace('.jpg', ''))
# Sort the images by integer index
images = sorted(images, key=lambda x: float((os.path.split(x)[1]).replace('image', '').replace('.jpg', '')))
print(images)
outvid = os.path.join('data/video', "outvideo.mp4")
make_video(outvid, images, fps=30)