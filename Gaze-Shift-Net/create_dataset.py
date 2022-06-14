from pycocotools.coco import COCO
import numpy as np
import argparse
import os
import sys
import pathlib as pl
from PIL import Image
import tensorflow as tf
import tensorflow.keras.backend as K
import os
import cv2
import random
from tqdm import tqdm
import argparse
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import resnet
import decoder
import SaliconLoader
from skimage import filters
import skimage.io as sio
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

np.random.seed(0)

parser = argparse.ArgumentParser()

parser.add_argument('--image_model_path', default = "./backbone/res_imagenet.pth", type=str, help='the path of the pre-trained model based on ImageNet')
parser.add_argument('--place_model_path', default = "./backbone/res_places.pth", type=str, help='the path of the pre-trained model based on PLACE')
parser.add_argument('--decoder_model_path', default = "./backbone/res_decoder.pth", type=str, help='the path of the pre-trained decoder model')
parser.add_argument('--gpu', default='0', type=str, help='The index of the gpu you want to use')
parser.add_argument('--size', default=(240 * 2, 320 * 2), type=tuple, help='resize the input image, (640,480) is from the training data, SALICON.')
parser.add_argument('--num_feat', default=5, type=int, help='the number of features collected from each model')
parser.add_argument('--path', default="./datasets/COCO", help='the path that contains raw coco JPEG images')
parser.add_argument('--salmodelpath', default="./saliency-shared-dir/mdsem_model_LSUN_Oct2019/mdsem_model_LSUN_full.h5", help='path for the saliency model')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


def normalize(x):
    x -= x.min()
    x /= x.max()

def post_process(pred):
    pred = filters.gaussian(pred, 5)
    # normalize(pred)
    # pred = (pred * 255).astype(np.uint8)
    return pred

saliency_model_path = args.salmodelpath
dataDir = args.path


img_model = resnet.resnet50(args.image_model_path).cuda().eval()
pla_model = resnet.resnet50(args.place_model_path).cuda().eval()
decoder_model = decoder.build_decoder(args.decoder_model_path, args.size, args.num_feat, args.num_feat).cuda().eval()

print("Saliency Models Loaded Successfully.")


mask_dir_inc = './datasets/CoCoClutter/increase/maskdir'
if not os.path.exists(mask_dir_inc):
    os.makedirs(mask_dir_inc)
seg_dir_inc = './datasets/CoCoClutter/increase/segdir'
if not os.path.exists(seg_dir_inc):
    os.makedirs(seg_dir_inc)


annFile = "/home/aj32632/fiftyone/coco-2017/raw/instances_train2017.json"
# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms_sc = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms_sc)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=nms)
nb_cats = max(catIds)
# map_cats = (((np.array(catIds)+100)/nb_cats) *255).astype(np.int).tolist()
map_cats = (np.array(catIds) + (255 - nb_cats)).tolist()

segDict = dict(zip(catIds, map_cats))
imgIds=[]
for i in catIds:
    imgIds += coco.getImgIds(catIds=i)
imgIds = list(dict.fromkeys(imgIds))
nb_images = len(imgIds)
size = 320, 240
bsize = size[0]*2, size[1]*2
counter = 0


# while counter < nb_images:
preprocess =    transforms.Compose([
                transforms.Resize(args.size),
	            transforms.ToTensor(),
    ])

for counter in tqdm(range(0,10), desc='image'):
    imId = imgIds.pop(np.random.randint(0, len(imgIds)))
    img = coco.loadImgs(imId)[0]
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    counteri = 0
    if len(anns) < 3:
        continue
    name = str(anns[0]['image_id'])
    namei = []
    maski = []
    for i in range(len(anns)):
        mask = coco.annToMask(anns[i])
        if i==0:
            insseg = np.round(mask) * anns[i]['category_id'] #segDict[anns[i]['category_id']]
        else:
            insseg += np.round(mask) * anns[i]['category_id'] #segDict[anns[i]['category_id']]
        ratio = mask.sum()/np.size(mask)
        if (ratio > 0.4):
            continue
        if (ratio < 0.03):
            continue

        minim = cv2.resize(mask, bsize, interpolation=cv2.INTER_NEAREST)
        im = Image.open(os.path.join("/home/aj32632/fiftyone/coco-2017/train/data", img['file_name']))
        if im.mode != 'RGB':
            continue
        
        im = np.array(im.resize(size))
        im = im[None, :, :, :]
        im[:, :, :, 0] = im[:, :, :, 0] - 103.939
        im[:, :, :, 1] = im[:, :, :, 1] - 116.779
        im[:, :, :, 2] = im[:, :, :, 2] - 123.68
        
        processed = preprocess(Image.fromarray(np.uint8(im[0]))).unsqueeze(0).cuda()
        with torch.no_grad():
            img_feat = img_model(processed, decode=True)
            pla_feat = pla_model(processed, decode=True)
            salmap = decoder_model([img_feat, pla_feat])
        
        # salmap = model.predict(im)[0, 0, :, :, 0]
        salmap = salmap.squeeze().detach().cpu().numpy()
        salmap = (salmap - np.min(salmap)) / (np.max(salmap) - np.min(salmap))
        if np.mean(salmap * minim) * (np.size(minim) / np.sum(minim)) > 0.7:
            continue

        namei.append('0'*(12 - len(name)) + name + '_%d.jpg'%counteri)
        maski.append(mask*255)
        counteri += 1

    if len(maski)>1:
        pick = random.randint(0, len(maski)-1)
        Image.fromarray(maski[pick]).save(os.path.join(mask_dir_inc, namei[pick]))
        # Image.fromarray(np.uint8(im[0])).save(os.path.join(mask_dir_inc, "_"+namei[pick]))
        counter +=1
        Image.fromarray(insseg).save(os.path.join(seg_dir_inc, '0'*(12 - len(name)) + name +'.jpg'))

