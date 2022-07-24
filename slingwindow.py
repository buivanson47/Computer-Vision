# import the necessary packages
import argparse
import time
import cv2
import imutils
import random
from matplotlib import pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-itrain", "--image_train", required=True,
	help="path to the input image train")
ap.add_argument("-iquery", "--image_query", required=True,
	help="path to the input image query")
args = vars(ap.parse_args())

# resize image
def pyramid(image, scale=1.2, minSize=(200, 200)):
	yield image
	while True:
		w = int(image.shape[1] / scale)
		# w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		yield image

# run sling window
def sliding_window(image, stepSize, windowSize):
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# load the input image
image_train = cv2.imread(args["image_train"]) # train image
image_query = cv2.imread(args["image_query"]) # query image


# define size window
winH, winW = image_query.shape[:-1]

# convert image to grayscale
train = cv2.cvtColor(image_train, cv2.COLOR_BGR2GRAY)
query = cv2.cvtColor(image_query, cv2.COLOR_BGR2GRAY)


# create sift and detect
MIN_MATCH_COUNT = 10
MIN_KEYPOINT = 10
FLANN_INDEX_KDTREE = 1
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(query, None)
print('[INFO] Keypoint of image query: ', len(kp1))

start = time.time() # define time start

print('[INFO] Start running sliding window...')
# for resized in pyramid(train, scale=1.2):

result = []
scores = []
number_window = 0
for (x, y, window) in sliding_window(train, stepSize=16, windowSize=(winW, winH)):
	if window.shape[0] != winH or window.shape[1] != winW:
		continue

	number_window += 1 # count number window
	clone = train.copy()
	regionProposal = clone[y:y+winH, x:x+winW]
	kp2, des2 = sift.detectAndCompute(regionProposal, None)

	if (len(kp2)>MIN_KEYPOINT):
		index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
		search_params = dict(checks=50)
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(des1, des2, k=2)
	
		good = []
		for m,n in matches:
			if m.distance < 0.75*n.distance:
				good.append(m)
			
		if(len(good)>0.3*len(kp1)):
			result.append((x,y,x+winW,y+winH))
			scores.append(len(good))


print('[INFO] Number window in train image: ', number_window)
print('[INFO] Number window have match good: ',len(result))

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

end = time.time() # define time end		
print('[INFO] Number region detected after filter iou: ',len(result_box))
print("[INFO] Selective search: {:.4f} seconds".format(end - start))

for rect in result_box:
    (xmin, ymin, xmax, ymax) = rect
    color = [random.randint(0, 255) for j in range(0, 3)]
    cv2.rectangle(image_train, (xmin, ymin), (xmax, ymax), color, 3)

plt.imshow(image_train[:,:,::-1]),plt.show()






