import argparse
import random
import time
import cv2
from matplotlib import pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-itrain", "--image_train", required=True,
	help="path to the input image train")
ap.add_argument("-iquery", "--image_query", required=True,
	help="path to the input image query")
ap.add_argument("-m", "--method", type=str, default="fast",
	choices=["fast", "quality"],
	help="selective search method")
args = vars(ap.parse_args())


# load the input image
image_train = cv2.imread(args["image_train"]) # train image
image_query = cv2.imread(args["image_query"]) # query image

# initialize OpenCV's selective search implementation and set the input image
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image_train)
# check to see if we are using the *fast* but *less accurate* version nof selective search
if args["method"] == "fast":
	print("[INFO] Using *fast* selective search")
	ss.switchToSelectiveSearchFast()
# otherwise we are using the *slower* but *more accurate* version
else:
	print("[INFO] Using *quality* selective search")
	ss.switchToSelectiveSearchQuality()

start = time.time()
rects = ss.process()
end = time.time()
print("[INFO] Selective search: {:.4f} seconds".format(end - start))
print("[INFO] {} total region proposals".format(len(rects)))

# filter rects
filter_rects = []
for rect in rects:
    (x, y, w, h) = rect
    h0, w0 = image_query.shape[:-1]
    if (w>0.45*w0) and (h>0.45*h0):
        filter_rects.append(rect)
print('[INFO] Number region after fillter size: ', len(filter_rects))


# detect keypoint in image query
MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 1
query = cv2.cvtColor(image_query, cv2.COLOR_BGR2GRAY) # query image scalegray
train = cv2.cvtColor(image_train, cv2.COLOR_BGR2GRAY) # train image scalegray

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(query, None)
print('[INFO] Keypoint of query image: ', len(kp1))

# detect keypoint in image filter rects and matching with query image
result = [] # region have len good better
scores = [] # store len(good) of region
for rect in filter_rects:
    (x, y, w, h) = rect
    regionProposal = train[y:y+h, x:x+w]
    # print(regionProposal.shape)
    
    kp2, des2 = sift.detectAndCompute(regionProposal, None)
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    if (len(kp2)>10):
        matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    if (len(good) > 0.3*len(kp1)):
        # print(len(good))
        result.append((x, y, x+w, y+h))
        scores.append(len(good))

print('[INFO] Detected number of region: ', len(result))

# caculate iou
def bb_intersection_over_union(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

# filter region have iou > 0.2
result_box = []
while(1):
    box = []
    sc = []

    max_score = max(scores)
    # print(max_score)
    true_box = result[scores.index(max_score)]
    # print(true_box)
    result_box.append(true_box)
    result.remove(true_box)
    scores.remove(max_score)
    for i in range(len(result)):
        iou = bb_intersection_over_union(list(result[i]), list(true_box))
        if (iou>=0.2):
        # add
            box.append(result[i])
            sc.append(scores[i])

    for i in range(len(sc)):
        result.remove(box[i])
        scores.remove(sc[i])

    # print(len(result))
    if (len(result)<1):
        break

print('[INFO] Number region detected after filter iou: ',len(result_box))
# image = cv2.imread('train_chinsu.jpg')
for rect in result_box:
    (xmin, ymin, xmax, ymax) = rect
    color = [random.randint(0, 255) for j in range(0, 3)]
    cv2.rectangle(image_train, (xmin, ymin), (xmax, ymax), color, 3)

plt.imshow(image_train[:,:,::-1]),plt.show()
