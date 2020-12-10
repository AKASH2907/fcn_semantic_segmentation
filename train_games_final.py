import datetime
import os
import random
import argparse
import torch

import numpy as np
import torchvision.transforms as transforms

import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.nn as nn
from torch import optim
# from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from PIL import Image
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from datasets import games_data
from fcn import VGGNet, FCN8s

from utils import evaluate, AverageMeter
from torchvision.utils import save_image
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/akumar/codes/cap_project/pytorch-semantic-segmentation/')

# cudnn.benchmark = True

ckpt_path = './games_ckpt'
exp_name = 'fcn8_512_1024'
writer = SummaryWriter(os.path.join(ckpt_path, 'exp', exp_name))

best_args = {
	'val_save_to_img_file': False,
	'val_img_sample_rate': 0.05  # randomly sample some validation results to display
}


def parse_args():
	parser = argparse.ArgumentParser(description='Games Semantic Segmentation FCN8')
	parser.add_argument('--gpu', type=str, default='0,1', help='gpu id')
	parser.add_argument('--epochs', type=int, default=50, help='number of rpochs to run')
	parser.add_argument('--seed', type=int, default=47, help='seed for training')
	parser.add_argument('--input_size', type=int, default=(512, 1024), help='val input image size')
	parser.add_argument('--crop_size', type=int, default=512, help='val input image size')
	parser.add_argument('--training_batch_size', type=int, default=16, help='training batch size')
	parser.add_argument('--val_batch_size', type=int, default=16, help='validation batch size')
	parser.add_argument('--lr_patience', type=int, default=2, help='learning rate patience')
	parser.add_argument('--snapshot', type=str, default='', help='weights path')
	parser.add_argument('--print_frequency', type=int, default=40, help='printing frequency')

	args = parser.parse_args()
	return args


def exist_directory(dir_path):
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)


def train(train_loader, model, device, criterion, optimizer, epoch, train_args):
	model.train()

	train_loss = AverageMeter()
	curr_iter = (epoch - 1) * len(train_loader)

	targets_all, preds_all = [], []

	for i, data in enumerate(train_loader):
		inputs, targets = data

		N = inputs.size(0)

		inputs, targets = inputs.to(device), targets.to(device)

		optimizer.zero_grad()
		outputs = model(inputs)
		assert outputs.size()[2:] == targets.size()[1:]
		assert outputs.size()[1] == games_data.num_classes

		loss = criterion(outputs, targets) / N
		loss.backward()
		optimizer.step()

		train_loss.update(loss.data, N)

		targets_all.append(targets.data.cpu().numpy())
		preds_all.append(outputs.data.max(1)[1].squeeze_(1).cpu().numpy())

		curr_iter += 1
		writer.add_scalar('train_loss', train_loss.avg, curr_iter)

		if (i + 1) % train_args.print_frequency == 0:
			print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (
				epoch, i + 1, len(train_loader), train_loss.avg))

	targets_all = np.concatenate(targets_all)
	preds_all = np.concatenate(preds_all)

	acc, acc_cls, mean_iou, _ = evaluate(preds_all, targets_all, games_data.num_classes)
	print('-----------------------------------------------------------------------------------------------------------')
	print('[epoch %d], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f]' % (
		epoch, acc, acc_cls, mean_iou))


def validate(val_loader, model, device, criterion, optimizer, epoch, train_args, restore, visualize):
	model.eval()

	val_loss = AverageMeter()
	inputs_all, gts_all, predictions_all = [], [], []

	with torch.no_grad():

		for vi, data in enumerate(val_loader):
			inputs, gts = data
			N = inputs.size(0)

			inputs, gts = inputs.to(device), gts.to(device)

			outputs = model(inputs)
			predictions = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()

			val_loss.update(criterion(outputs, gts).data / N, N)

			for i in inputs:
				if random.random() > train_args['val_img_sample_rate']:
					inputs_all.append(None)
				else:
					inputs_all.append(i.data.cpu())
			gts_all.append(gts.data.cpu().numpy())
			predictions_all.append(predictions)

	gts_all = np.concatenate(gts_all)
	predictions_all = np.concatenate(predictions_all)

	acc, acc_cls, mean_iu, _ = evaluate(predictions_all, gts_all, games_data.num_classes)


	if mean_iu > train_args['best_record']['mean_iu']:
		train_args['best_record']['val_loss'] = val_loss.avg
		train_args['best_record']['epoch'] = epoch
		train_args['best_record']['acc'] = acc
		train_args['best_record']['acc_cls'] = acc_cls
		train_args['best_record']['mean_iu'] = mean_iu
		snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f' % (
			epoch, val_loss.avg, acc, acc_cls, mean_iu
		)
		torch.save(model.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))

		if train_args['val_save_to_img_file']:
			to_save_dir = os.path.join(ckpt_path, exp_name, str(epoch))
			exist_directory(to_save_dir)

		val_visual = []
		for idx, data in enumerate(zip(inputs_all, gts_all, predictions_all)):
			if data[0] is None:
				continue
			input_pil = restore(data[0])
			gt_pil = games_data.colorize_mask(data[1])
			predictions_pil = games_data.colorize_mask(data[2])
			if train_args['val_save_to_img_file']:
				input_pil.save(os.path.join(to_save_dir, '%d_input.png' % idx))
				predictions_pil.save(os.path.join(to_save_dir, '%d_prediction.png' % idx))
				gt_pil.save(os.path.join(to_save_dir, '%d_gt.png' % idx))
			val_visual.extend([visualize(input_pil.convert('RGB')), visualize(gt_pil.convert('RGB')),
							   visualize(predictions_pil.convert('RGB'))])
		val_visual = torch.stack(val_visual, 0)
		val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)
		writer.add_image(snapshot_name, val_visual)

	print('-----------------------------------------------------------------------------------------------------------')
	print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f]' % (
		epoch, val_loss.avg, acc, acc_cls, mean_iu))

	print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [epoch %d]' % (
		train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
		train_args['best_record']['mean_iu'], train_args['best_record']['epoch']))
	print('-----------------------------------------------------------------------------------------------------------')

	writer.add_scalar('val_loss', val_loss.avg, epoch)
	writer.add_scalar('acc', acc, epoch)
	writer.add_scalar('acc_cls', acc_cls, epoch)
	writer.add_scalar('mean_iu', mean_iu, epoch)

	model.train()
	return val_loss.avg

def main():
	args = parse_args()

	torch.backends.cudnn.benchmark = True
	os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
	device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


	# Random seed for reproducibility
	if args.seed:
		random.seed(args.seed)
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		if args.gpu:
			torch.cuda.manual_seed_all(args.seed)
	
	# seed = 63
	# random.seed(seed)
	# np.random.seed(seed)
	# torch.manual_seed(seed)
	# # if args.gpu:
	# torch.cuda.manual_seed_all(seed)
	
	denoramlize_argument = ([0.485, 0.456, 0.406],
							[0.229, 0.224, 0.225])

	train_transforms = transforms.Compose([
		transforms.RandomCrop((args.crop_size, args.crop_size)),
		transforms.RandomRotation(90),
		transforms.RandomHorizontalFlip(p=0.5),
		])

	# train_joint_transform = joint_transforms.Compose([
	# 	# joint_transforms.Scale(img_resize_shape),
	# 	joint_transforms.RandomCrop(args['crop_size']),
	# 	joint_transforms.RandomHorizontallyFlip(),
	# 	joint_transforms.RandomRotate(90)
	# ])

	img_resize_shape = int(min(args.input_size) / 0.8)
	# val_transforms = transforms.Compose([
	# 	transforms.Scale(img_resize_shape, interpolation=Image.NEAREST),
	# 	transforms.CenterCrop(args['input_size'])
	# 	])
	
	val_joint_transform = joint_transforms.Compose([
		joint_transforms.Scale(img_resize_shape),
		joint_transforms.CenterCrop(args.input_size)
	])
	input_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406],
							 [0.229, 0.224, 0.225])
	])
	target_transform = extended_transforms.MaskToTensor()
	restore_transform = transforms.Compose([
		extended_transforms.DeNormalize(*denoramlize_argument),
		transforms.ToPILImage()
	])
	visualize = transforms.ToTensor()

	# train_set = games_data.CityScapes('train', joint_transform=train_joint_transform,
	# 								  transform=input_transform, target_transform=target_transform)
	train_set = games_data.CityScapes('train', transform=train_transforms)
	# train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)
	train_loader = DataLoader(train_set, batch_size=args.training_batch_size, num_workers=8, shuffle=True)
	val_set = games_data.CityScapes('val', joint_transform=val_joint_transform, transform=input_transform,
									target_transform=target_transform)
	val_loader = DataLoader(val_set, batch_size=args.val_batch_size, num_workers=8, shuffle=True)
	
	print(len(train_loader), len(val_loader))
	# sdf

	
	
	# Load pretrained VGG model
	vgg_model = VGGNet(requires_grad=True, remove_fc=True)
	
	# FCN architecture load
	model = FCN8s(pretrained_net=vgg_model, n_class=games_data.num_classes, dropout_rate=0.4)

	# Loss function
	criterion = nn.CrossEntropyLoss(ignore_index=games_data.ignore_label)

	# Optimizer
	optimizer = optim.Adam(model.parameters(), lr=1e-4)

	# Create directory for checkpoints
	exist_directory(ckpt_path)
	exist_directory(os.path.join(ckpt_path, exp_name))
	open(os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(args) + '\n\n')

	# Learning rate scheduler
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.lr_patience, min_lr=1e-10)

	# Send model to CUDA device
	vgg_model = vgg_model.to(device)
	model = model.to(device)

	# Use if more than 1 GPU
	if torch.cuda.device_count()>1:
		model = nn.DataParallel(model)
		

	if len(args.snapshot) == 0:
		curr_epoch = 1
		best_args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0}
	else:
		print('training resumes from ' + args['snapshot'])
		model.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'])))
		split_snapshot = args['snapshot'].split('_')
		curr_epoch = int(split_snapshot[1]) + 1
		best_args['best_record'] = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
							   'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
							   'mean_iu': float(split_snapshot[9][:-4])}

	criterion.to(device)

	for epoch in range(curr_epoch, args.epochs + 1):
		train(train_loader, model, device, criterion, optimizer, epoch, args)
		val_loss = validate(val_loader, model, device, criterion, optimizer, epoch, args, restore_transform, visualize)
		scheduler.step(val_loss)





if __name__ == '__main__':
	main()



# train_data = iter(train_loader)
# train_imgs, train_msks = next(train_data)
# print(train_imgs[0].shape, train_msks[0].shape)

# val_data = iter(val_loader)
# image, mask = next(val_data)
# print(image.shape, mask.shape)
# print(image[0].shape, mask[0].shape)
# sdfs
# # images_to_plot = 9
# fig = plt.figure()

# check for numpy array

# for i in range(16):
# 	train_img = train_imgs[i]
# 	train_msk = train_msks[i]
# 	img = image[i]
# 	msk = mask[i]
	
# 	train_img = train_img.numpy()
# 	train_msk = train_msk.numpy()
# 	img = image.numpy()
# 	msk = msk.numpy()

# 	# print(type(img), type(msk))
	
# 	unique = np.unique(train_img)
# 	print(len(unique))
# 	unique = np.unique(img)
# 	print(len(unique))
# 	un1 = np.unique(train_msk)
# 	print(un1)
# 	unique = np.unique(msk)
# 	print(unique)
# sdf

	# img_n.save('img_0' + str(i) + '.png')
	# msk_n.save('msk_0' + str(i) + '.png')
	# save_image(img, 'image_{}.png'.format(i))
	# save_image(msk.float(), 'mask_{}.png'.format(i))
# 	img = mask[i]
# 	# img = np.transpose(img, (1, 2, 0))
# 	# ax = fig.add_subplot(3, 3, i+1)
# 	# ax.imshow(img, cmap='gray')
# 	# plt.subplots(3, 3, i)
# 	plt.imshow(img, cmap='gray')
# 	plt.savefig("mask" + str(i+1) + ".png")
# 	# plt.axis('off')
# # fig.savefig('masks.png')
# # plt.savefig('images.png')
# for i, batch in enumerate(train_loader, start=1):
# 	image, mask = batch

# 	for x,y in :
# 		img = 

# # 	plt.subplot(3, 3, i)
# # 	plt.imshow(image)
# # 	plt.axis('off')
# # 	# plt.title(train_set.classes[label.item()], fontsize=28)
# # 	if (i >= how_many_to_plot): break
# # figs, axes = plt.subplots(3, 3)