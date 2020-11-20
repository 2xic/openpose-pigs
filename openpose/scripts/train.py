import argparse
import os
import time

import cv2
import matplotlib
import numpy as np
import torch
import torch.optim as optim
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

matplotlib.use('Agg')

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def timeout(strt, start):
	if (strt * 60) < (time.time() - start):
		return True
	return False

def train(prepared_train_labels, train_images_folder, num_refinement_stages, base_lr, batch_size, batches_per_iter,
		  num_workers, checkpoint_path, weights_only, from_mobilenet, checkpoints_folder, log_after,
		  val_labels, val_images_folder, val_output_name, checkpoint_after, val_after):

	KEY_POINTS = 6
	net = PoseEstimationWithMobileNet(num_refinement_stages, num_heatmaps=(KEY_POINTS + 1), num_pafs=((KEY_POINTS - 1) * 2))

	stride = 8
	sigma = 7
	path_thickness = 1
	transformer = transforms.Compose([
		Scale(),
		Rotate(pad=(128, 128, 128)),
		CropPad(pad=(128, 128, 128))
	])

	dataset = CocoTrainDataset(prepared_train_labels, train_images_folder,
							   stride, sigma, path_thickness, transform=transformer)
	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

	optimizer = optim.Adam([
		{'params': get_parameters_conv(net.model, 'weight')},
		{'params': get_parameters_conv_depthwise(net.model, 'weight'), 'weight_decay': 0},
		{'params': get_parameters_bn(net.model, 'weight'), 'weight_decay': 0},
		{'params': get_parameters_bn(net.model, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
		{'params': get_parameters_conv(net.cpm, 'weight'), 'lr': base_lr},
		{'params': get_parameters_conv(net.cpm, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
		{'params': get_parameters_conv_depthwise(net.cpm, 'weight'), 'weight_decay': 0},
		{'params': get_parameters_conv(net.initial_stage, 'weight'), 'lr': base_lr},
		{'params': get_parameters_conv(net.initial_stage, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
		{'params': get_parameters_conv(net.refinement_stages, 'weight'), 'lr': base_lr * 4},
		{'params': get_parameters_conv(net.refinement_stages, 'bias'), 'lr': base_lr * 8, 'weight_decay': 0},
		{'params': get_parameters_bn(net.refinement_stages, 'weight'), 'weight_decay': 0},
		{'params': get_parameters_bn(net.refinement_stages, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
	], lr=base_lr, weight_decay=5e-4)

	num_iter = 0
	current_epoch = 0
	drop_after_epoch = [100, 200, 260]
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.333)
	if os.path.isfile("model"):
		load_from_mobilenet(net, torch.load("model"))

	net = DataParallel(net).cuda()
	net.train()
	minutes_before_timeout = 60 * 6

	start = time.time()
	batch_loss = []

	torch.autograd.set_detect_anomaly(True)
	while not timeout(minutes_before_timeout, start):
		if epochId in drop_after_epoch:
			scheduler.step()
		total_losses = [0, 0] * (num_refinement_stages + 1)  # heatmaps loss, paf loss per stage
		batch_per_iter_idx = 0
		if timeout(minutes_before_timeout, start):
			break
		for batch_data in train_loader:
			if batch_per_iter_idx == 0:
				optimizer.zero_grad()
			if timeout(minutes_before_timeout, start):
				break
			images = batch_data['image'].cuda()
			keypoint_masks = batch_data['keypoint_mask'].cuda()
			paf_masks = batch_data['paf_mask'].cuda()
			keypoint_maps = batch_data['keypoint_maps'].cuda()
			paf_maps = batch_data['paf_maps'].cuda()

			stages_output = net(images)

			losses = []
			for loss_idx in range(len(total_losses) // 2):
				losses.append(l2_loss(stages_output[loss_idx * 2], keypoint_maps, keypoint_masks, images.shape[0]))
				losses.append(l2_loss(stages_output[loss_idx * 2 + 1], paf_maps, paf_masks, images.shape[0]))
				total_losses[loss_idx * 2] += losses[-2].item() / batches_per_iter
				total_losses[loss_idx * 2 + 1] += losses[-1].item() / batches_per_iter

			loss = losses[0]
			for loss_idx in range(1, len(losses)):
				loss += losses[loss_idx]

			batch_loss.append(loss.item())
			loss.backward()
			batch_per_iter_idx += 1
			optimizer.step()
			batch_per_iter_idx = 0
			num_iter += 1

			if num_iter % log_after == 0:
				print('Iter: {}'.format(num_iter))
				for loss_idx in range(len(total_losses) // 2):
					print('\n'.join(['stage{}_pafs_loss:     {}', 'stage{}_heatmaps_loss: {}']).format(
						loss_idx + 1, total_losses[loss_idx * 2 + 1] / log_after,
						loss_idx + 1, total_losses[loss_idx * 2] / log_after))

				for loss_idx in range(len(total_losses)):
					total_losses[loss_idx] = 0

	snapshot_name = "model_{}".format(int(time.time()))
	torch.save({'state_dict': net.module.state_dict(),
		'optimizer': optimizer.state_dict(),
		'scheduler': scheduler.state_dict(),
		'iter': num_iter,
		'current_epoch': epochId},
			snapshot_name)
	print(batch_loss)
	return net

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-train-labels', type=str, required=True,
                        help='path to the file with prepared annotations')
    parser.add_argument('--train-images-folder', type=str, required=True, help='path to COCO train images folder')
	parser.add_argument('--num-refinement-stages', type=int, default=1, help='number of refinement stages')
	parser.add_argument('--base-lr', type=float, default=3e-4, help='initial learning rate')
	parser.add_argument('--batch-size', type=int, default=1, help='batch size')
	parser.add_argument('--batches-per-iter', type=int, default=1, help='number of batches to accumulate gradient from')
	parser.add_argument('--num-workers', type=int, default=2, help='number of workers')
	parser.add_argument('--checkpoint-path', type=str, required=False, help='path to the checkpoint to continue training from')
	parser.add_argument('--from-mobilenet', action='store_true',
						help='load weights from mobilenet feature extractor')
	parser.add_argument('--weights-only', action='store_true',
						help='just initialize layers with pre-trained weights and start training from the beginning')
	parser.add_argument('--experiment-name', type=str, default='default',
						help='experiment name to create folder for checkpoints')
	parser.add_argument('--log-after', type=int, default=100, help='number of iterations to print train loss')

	parser.add_argument('--val-labels', type=str, required=False, help='path to json with keypoints val labels')
	parser.add_argument('--val-images-folder', type=str, required=False, help='path to COCO val images folder')
	parser.add_argument('--val-output-name', type=str, default='detections.json',
						help='name of output json file with detected keypoints')
	parser.add_argument('--checkpoint-after', type=int, default=5000,
						help='number of iterations to save checkpoint')
	parser.add_argument('--val-after', type=int, default=5000,
						help='number of iterations to run validation')
	args = parser.parse_args()


	net_work = train(args.prepared_train_labels, args.train_images_folder,args.num_refinement_stages, args.base_lr, args.batch_size,
		  args.batches_per_iter, args.num_workers, args.checkpoint_path, args.weights_only, args.from_mobilenet,
		  None, args.log_after, args.val_labels, args.val_images_folder, args.val_output_name,
		  args.checkpoint_after, args.val_after)

