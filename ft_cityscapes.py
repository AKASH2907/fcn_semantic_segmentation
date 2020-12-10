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
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from PIL import Image
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from datasets import cityscapes
from fcn import VGGNet, FCN8s, FCNs

from utils import check_mkdir, evaluate, AverageMeter
from torchvision.utils import save_image
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/akumar/codes/cap_project/pytorch-semantic-segmentation/')

cudnn.benchmark = True

ckpt_path = './games_ckpt'
ft_ckpt_path = '../cityscapes_ckpt'
# exp_name = 'games-fcn8s-bs16'
exp_name = 'fcn8_1024'
ft_exp_name = 'ft_1024'
writer = SummaryWriter(os.path.join(ckpt_path, 'exp', exp_name))

args = {
	'train_batch_size': 16,
	'epoch_num': 80,
	'lr': 1e-4,
	'weight_decay': 5e-4,
	'input_size': (512, 1024),
	'crop_size': (512, 512),
	'momentum': 0.95,
	'lr_patience': 2,  # large patience denotes fixed lr
	# 'snapshot': 'epoch_76_loss_0.01034_acc_0.94963_acc-cls_0.36807_mean-iu_0.29726_fwavacc_0.91834.pth',  # empty string denotes no snapshot
	'snapshot':'epoch_42_loss_0.00916_acc_0.95598_acc-cls_0.58651_mean-iu_0.50990.pth',
	'print_freq': 30,
	'val_batch_size': 16,
	'val_save_to_img_file': False,
	'val_img_sample_rate': 0.05  # randomly sample some validation results to display
}


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
#         torch.nn.init.xavier_uniform(m.bias.data, 0)
        nn.init.constant_(m.bias, 0)


def parse_args():
	parser = argparse.ArgumentParser(description='Games Semantic Segmentation FCN8')
	parser.add_argument('--gpu', type=str, default='0,1', help='gpu id')
	parser.add_argument('--epochs', type=int, default=50, help='number of rpochs to run')
	parser.add_argument('--seed', type=int, default=47, help='seed for training')
	args = parser.parse_args()
	return args


def main():
	# args = parse_args()

	torch.backends.cudnn.benchmark = True
	os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
	device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


	

	# # if args.seed:
	# random.seed(args.seed)
	# np.random.seed(args.seed)
	# torch.manual_seed(args.seed)
	# # if args.gpu:
	# torch.cuda.manual_seed_all(args.seed)
	seed = 63
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	# if args.gpu:
	torch.cuda.manual_seed_all(seed)
	

	mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	# train_transforms = transforms.Compose([
	# 	transforms.RandomCrop(args['crop_size']),
	# 	transforms.RandomRotation(90),
	# 	transforms.RandomHorizontalFlip(p=0.5),
	# 	transforms.RandomVerticalFlip(p=0.5),
		
	# 	])
	short_size = int(min(args['input_size']) / 0.875)
	# val_transforms = transforms.Compose([
	# 	transforms.Scale(short_size, interpolation=Image.NEAREST),
	# 	# joint_transforms.Scale(short_size),
	# 	transforms.CenterCrop(args['input_size'])
	# 	])
	train_joint_transform = joint_transforms.Compose([
		# joint_transforms.Scale(short_size),
		joint_transforms.RandomCrop(args['crop_size']),
		joint_transforms.RandomHorizontallyFlip(),
		joint_transforms.RandomRotate(90)
	])
	val_joint_transform = joint_transforms.Compose([
		joint_transforms.Scale(short_size),
		joint_transforms.CenterCrop(args['input_size'])
	])
	input_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(*mean_std)
	])
	target_transform = extended_transforms.MaskToTensor()
	restore_transform = transforms.Compose([
		extended_transforms.DeNormalize(*mean_std),
		transforms.ToPILImage()
	])
	visualize = transforms.ToTensor()

	train_set = cityscapes.CityScapes('train', joint_transform=train_joint_transform,
									  transform=input_transform, target_transform=target_transform)
	# train_set = cityscapes.CityScapes('train', transform=train_transforms)
	train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)
	val_set = cityscapes.CityScapes('val', joint_transform=val_joint_transform, transform=input_transform,
									target_transform=target_transform)
	# val_set = cityscapes.CityScapes('val', transform=val_transforms)
	val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=8, shuffle=True)
	
	print(len(train_loader), len(val_loader))
	
	# sdf
	
	vgg_model = VGGNet(requires_grad=True, remove_fc=True)
	net = FCN8s(pretrained_net=vgg_model, n_class=cityscapes.num_classes, dropout_rate=0.4)
	# net.apply(init_weights)
	criterion = nn.CrossEntropyLoss(ignore_index=cityscapes.ignore_label)

	optimizer = optim.Adam(net.parameters(), lr=1e-4)


	check_mkdir(ft_ckpt_path)
	check_mkdir(os.path.join(ft_ckpt_path, ft_exp_name))
	open(os.path.join(ft_ckpt_path, ft_exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(args) + '\n\n')

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args['lr_patience'], min_lr=1e-10)

	vgg_model = vgg_model.to(device)
	net = net.to(device)

	if torch.cuda.device_count()>1:
		net = nn.DataParallel(net)
		

	# if len(args['snapshot']) == 0:
	# 	curr_epoch = 1
	# 	args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0}
	# else:
	# print('training resumes from ' + args['snapshot'])
	net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'])))
	# split_snapshot = args['snapshot'].split('_')
	# curr_epoch = int(split_snapshot[1]) + 1
	curr_epoch = 1
	args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0}
	# args['best_record'] = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
	# 					   'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
	# 					   'mean_iu': float(split_snapshot[9][:-4])}

	criterion.to(device)

	for epoch in range(curr_epoch, args['epoch_num'] + 1):
		train(train_loader, net, device, criterion, optimizer, epoch, args)
		val_loss = validate(val_loader, net, device, criterion, optimizer, epoch, args, restore_transform, visualize)
		scheduler.step(val_loss)


def train(train_loader, net, device, criterion, optimizer, epoch, train_args):
	net.train()

	train_loss = AverageMeter()
	curr_iter = (epoch - 1) * len(train_loader)

	targets_all, preds_all = [], []

	for i, data in enumerate(train_loader):
		inputs, targets = data

		assert inputs.size()[2:] == targets.size()[1:]
		N = inputs.size(0)

		inputs, targets = inputs.to(device), targets.to(device)

		optimizer.zero_grad()
		outputs = net(inputs)
		assert outputs.size()[2:] == targets.size()[1:]
		assert outputs.size()[1] == cityscapes.num_classes

		loss = criterion(outputs, targets) / N
		loss.backward()
		optimizer.step()

		train_loss.update(loss.data, N)

		targets_all.append(targets.data.cpu().numpy())
		preds_all.append(outputs.data.max(1)[1].squeeze_(1).cpu().numpy())

		curr_iter += 1
		writer.add_scalar('train_loss', train_loss.avg, curr_iter)

		if (i + 1) % train_args['print_freq'] == 0:
			print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (
				epoch, i + 1, len(train_loader), train_loss.avg))

	targets_all = np.concatenate(targets_all)
	preds_all = np.concatenate(preds_all)

	acc, acc_cls, mean_iou, _ = evaluate(preds_all, targets_all, cityscapes.num_classes)
	print('-----------------------------------------------------------------------------------------------------------')
	print('[epoch %d], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f]' % (
		epoch, acc, acc_cls, mean_iou))


def validate(val_loader, net, device, criterion, optimizer, epoch, train_args, restore, visualize, finetuning=True):
	net.eval()

	val_loss = AverageMeter()
	inputs_all, gts_all, predictions_all = [], [], []

	with torch.no_grad():

		for vi, data in enumerate(val_loader):
			inputs, gts = data
			N = inputs.size(0)

			inputs, gts = inputs.to(device), gts.to(device)

			outputs = net(inputs)
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

	acc, acc_cls, mean_iu, _ = evaluate(predictions_all, gts_all, cityscapes.num_classes)


	if mean_iu > train_args['best_record']['mean_iu']:
		train_args['best_record']['val_loss'] = val_loss.avg
		train_args['best_record']['epoch'] = epoch
		train_args['best_record']['acc'] = acc
		train_args['best_record']['acc_cls'] = acc_cls
		train_args['best_record']['mean_iu'] = mean_iu
		snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f' % (
			epoch, val_loss.avg, acc, acc_cls, mean_iu
		)
		if finetuning == False:
			
			torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))

			if train_args['val_save_to_img_file']:
				to_save_dir = os.path.join(ckpt_path, exp_name, str(epoch))
				check_mkdir(to_save_dir)

			val_visual = []
			for idx, data in enumerate(zip(inputs_all, gts_all, predictions_all)):
				if data[0] is None:
					continue
				input_pil = restore(data[0])
				gt_pil = cityscapes.colorize_mask(data[1])
				predictions_pil = cityscapes.colorize_mask(data[2])
				if train_args['val_save_to_img_file']:
					input_pil.save(os.path.join(to_save_dir, '%d_input.png' % idx))
					predictions_pil.save(os.path.join(to_save_dir, '%d_prediction.png' % idx))
					gt_pil.save(os.path.join(to_save_dir, '%d_gt.png' % idx))
				val_visual.extend([visualize(input_pil.convert('RGB')), visualize(gt_pil.convert('RGB')),
								   visualize(predictions_pil.convert('RGB'))])
			val_visual = torch.stack(val_visual, 0)
			val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)
			writer.add_image(snapshot_name, val_visual)

		else:

			torch.save(net.state_dict(), os.path.join(ft_ckpt_path, ft_exp_name, snapshot_name + '.pth'))

			if train_args['val_save_to_img_file']:
				to_save_dir = os.path.join(ft_ckpt_path, ft_exp_name, str(epoch))
				check_mkdir(to_save_dir)

			val_visual = []
			for idx, data in enumerate(zip(inputs_all, gts_all, predictions_all)):
				if data[0] is None:
					continue
				input_pil = restore(data[0])
				gt_pil = cityscapes.colorize_mask(data[1])
				predictions_pil = cityscapes.colorize_mask(data[2])
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

	net.train()
	return val_loss.avg


if __name__ == '__main__':
	main()
