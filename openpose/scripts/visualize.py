import argparse
import os
import time

import cv2
import matplotlib
import numpy as np
import torch
import torch.optim as optim
from PIL import Image as PILImage
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import transforms

from openpose.datasets.coco import CocoTrainDataset
from openpose.datasets.transformations import (ConvertKeypoints, CropPad, Flip,
                                               Rotate, Scale)
from openpose.models.with_mobilenet import PoseEstimationWithMobileNet
from openpose.modules.get_parameters import (get_parameters_bn,
                                             get_parameters_conv,
                                             get_parameters_conv_depthwise)
from openpose.modules.keypoints import extract_keypoints, group_keypoints
from openpose.modules.load_state import load_from_mobilenet, load_state
from openpose.modules.loss import l2_loss
from openpose.modules.pose import Pose

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

if not os.path.isfile("model"):
	print("model file need to be in same as visualize")

parser = argparse.ArgumentParser()
parser.add_argument('--image-name', type=str, required=True, help='path to the image')
args = parser.parse_args()

image = cv2.imread(args.image_name, cv2.IMREAD_COLOR)
image = image.astype(np.float32)
image = (image - 128) / 256
image = image.transpose((2, 0, 1))

size = image.shape[1:]
KEY_POINTS = 6

model_image_input = torch.from_numpy(image.reshape((1, ) + image.shape))
net = PoseEstimationWithMobileNet(num_refinement_stages=1, num_heatmaps=(KEY_POINTS + 1), num_pafs=((KEY_POINTS - 1) * 2))
load_state(net, torch.load("model", map_location='cpu'))

if torch.cuda.is_available():
	model_image_input = model_image_input.cuda()
	net = net.cuda()

stages_output = net(model_image_input)

stage2_heatmaps =  stages_output[-2]
heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))

stage2_pafs = stages_output[-1]
pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))

total_keypoints_num = 0
all_keypoints_by_type = []
num_keypoints = Pose.num_kpts + 1

delta_y = size[0] / pafs.shape[0]
delta_x = size[1] / pafs.shape[1]

for kpt_idx in range(num_keypoints):  # 19th for bg
	C = heatmaps[:, :, kpt_idx]
	total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

conf = {
	"defined_success_ratio": 0.5,
	"point_score": 100
}
pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True, **conf)

img = cv2.imread(args.image_name, cv2.IMREAD_COLOR)

for index, entry_id in enumerate(all_keypoints_by_type):
	for keypoint in entry_id:
		x, y, _, _ = keypoint

		x = int(x * delta_x)
		y = int(y * delta_y)

		img = cv2.circle(img, (x, y), radius=15, color=(255, 255, 255), thickness=-1)
		img = cv2.putText(img, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


current_poses = []
for n in range(len(pose_entries)):
	if len(pose_entries[n]) == 0:
		continue
	pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
	for kpt_id in range(num_keypoints):
		if pose_entries[n][kpt_id] != -1.0:
			pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
			pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
	pose = Pose(pose_keypoints, pose_entries[n][18])
	current_poses.append(pose)

for pose in current_poses:
	pose.draw(img, scale_x=delta_x, scale_y=delta_y)

cv2.imwrite('output_{}.jpg'.format(os.path.basename(args.image_name)),img)
