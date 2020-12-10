import os
import torch
import numpy as np

import torch.nn as nn
import torchvision.transforms as transforms
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader

from datasets import cityscapes
from utils import check_mkdir, evaluate
from fcn import VGGNet, FCN8s, FCNs


cudnn.benchmark = True

ckpt_path = '../cityscapes_ckpt'

args = {
    'input_size': (512, 1024),
    'exp_name': 'ft_1024',
    'snapshot': 'epoch_18_loss_0.00797_acc_0.96262_acc-cls_0.48482_mean-iu_0.39980.pth'
}


def main():

    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


    vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    net = FCN8s(pretrained_net=vgg_model, n_class=cityscapes.num_classes, dropout_rate=0.4)
    print('load model ' + args['snapshot'])
    

    vgg_model = vgg_model.to(device)
    net = net.to(device)

    if torch.cuda.device_count()>1:
        net = nn.DataParallel(net)
    net.load_state_dict(torch.load(os.path.join(ckpt_path, args['exp_name'], args['snapshot'])))
    net.eval()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    short_size = int(min(args['input_size']) / 0.875)
    val_joint_transform = joint_transforms.Compose([
        joint_transforms.Scale(short_size),
        joint_transforms.CenterCrop(args['input_size'])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()
    restore_transform = transforms.Compose([
        extended_transforms.DeNormalize(*mean_std),
        transforms.ToPILImage()
    ])

    # test_set = cityscapes.CityScapes('test', transform=test_transform)

    test_set = cityscapes.CityScapes('test', joint_transform=val_joint_transform, transform=test_transform,
                                    target_transform=target_transform)

    test_loader = DataLoader(test_set, batch_size=1, num_workers=8, shuffle=False)
    
    transform = transforms.ToPILImage()

    check_mkdir(os.path.join(ckpt_path, args['exp_name'], 'test'))

    gts_all, predictions_all = [], []
    count = 0
    for vi, data in enumerate(test_loader):
        # img_name, img = data
        img_name, img, gts = data
        
        img_name = img_name[0]
        # print(img_name)
        img_name = img_name.split('/')[-1]
        # img.save(os.path.join(ckpt_path, args['exp_name'], 'test', img_name))

        img_transform = restore_transform(img[0])
        # img_transform = img_transform.convert('RGB')
        img_transform.save(os.path.join(ckpt_path, args['exp_name'], 'test', img_name))
        img_name = img_name.split('_leftImg8bit.png')[0]

        # img = Variable(img, volatile=True).cuda()
        img, gts = img.to(device), gts.to(device)
        output = net(img)

        prediction = output.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        prediction_img = cityscapes.colorize_mask(prediction)
        # print(type(prediction_img))
        prediction_img.save(os.path.join(ckpt_path, args['exp_name'], 'test', img_name + '.png'))
        # print(ckpt_path, args['exp_name'], 'test', img_name + '.png')

        print('%d / %d' % (vi + 1, len(test_loader)))
        gts_all.append(gts.data.cpu().numpy())
        predictions_all.append(prediction)
        # break

        # if count == 1: 
        #     break
        # count += 1 
    gts_all = np.concatenate(gts_all)
    predictions_all = np.concatenate(prediction)
    acc, acc_cls, mean_iou, _ = evaluate(predictions_all, gts_all, cityscapes.num_classes)


    print('-----------------------------------------------------------------------------------------------------------')
    print('[acc %.5f], [acc_cls %.5f], [mean_iu %.5f]' % (acc, acc_cls, mean_iu))


if __name__ == '__main__':
    main()
