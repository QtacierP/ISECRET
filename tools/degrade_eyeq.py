import numpy as np
import cv2
import random
import time
import math
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from matplotlib.pyplot import imread, imsave
import os
import sys
from PIL import Image
import argparse

args = argparse.ArgumentParser(description='the option of the degradation of EyeQ')

args.add_argument('--degrade_dir', type=str, default='', help='degrade EyeQ dir')
args.add_argument('--gt_dir', type=str, default='', help='high quality cropped image dir')
args.add_argument('--output_dir', type=str, default='./temp', help='degraded output dir')
args.add_argument('--mask_dir', type=str, default='./temp', help='mask output dir')


args = args.parse_args()
'''
===== Gen Mask ====
'''

def _get_center_radius_by_hough(mask):
    circles = cv2.HoughCircles((mask*255).astype(np.uint8),cv2.HOUGH_GRADIENT,1,1000,param1=5,param2=5,minRadius=min(mask.shape)//4, maxRadius=max(mask.shape)//2+1)
    center = circles[0,0,:2]
    radius = circles[0,0,2]
    return center, radius

def _get_circle_by_center_bbox(shape,center,bbox,radius):
    center_mask=np.zeros(shape=shape).astype('uint8')
    tmp_mask=np.zeros(shape=bbox[2:4])
    center_tmp=(int(center[0]),int(center[1]))
    center_mask=cv2.circle(center_mask,center_tmp[::-1],int(radius),(1),-1)
    return center_mask

def get_mask(img):
    if img.ndim ==3:
        g_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    elif img.ndim == 2:
        g_img = img.copy()
    else:
        raise RuntimeError
    h,w = g_img.shape
    shape=g_img.shape[0:2]
    g_img = cv2.resize(g_img,(0,0),fx = 0.5,fy = 0.5)
    tg_img=cv2.normalize(g_img, None, 0, 255, cv2.NORM_MINMAX)
    tmp_mask=get_mask_BZ(tg_img)
    center, radius = _get_center_radius_by_hough(tmp_mask)
    #resize back
    center = [center[1]*2,center[0]*2]
    radius = int(radius*2)
    s_h = max(0,int(center[0] - radius))
    s_w = max(0, int(center[1] - radius))
    bbox = (s_h, s_w, min(h-s_h,2 * radius), min(w-s_w,2 * radius))
    tmp_mask=_get_circle_by_center_bbox(shape,center,bbox,radius)
    return tmp_mask,bbox,center,radius

def mask_image(img,mask):
    img[mask<=0,...]=0
    return img

def supplemental_black_area(img,border=None):
    image=img
    if border is None:
        h,v=img.shape[0:2]
        max_l=max(h,v)
        if image.ndim>2:
            image=np.zeros(shape=[max_l,max_l,img.shape[2]],dtype=img.dtype)
        else:
            image=np.zeros(shape=[max_l,max_l],dtype=img.dtype)
        border=(int(max_l/2-h/2),int(max_l/2-h/2)+h,int(max_l/2-v/2),int(max_l/2-v/2)+v,max_l)
    else:
        max_l=border[4]
        if image.ndim>2:
            image=np.zeros(shape=[max_l,max_l,img.shape[2]],dtype=img.dtype)
        else:
            image=np.zeros(shape=[max_l,max_l],dtype=img.dtype)
    image[border[0]:border[1],border[2]:border[3],...]=img
    return image,border


def remove_back_area(img,bbox=None,border=None):
    image=img
    if border is None:
        border=np.array((bbox[0],bbox[0]+bbox[2],bbox[1],bbox[1]+bbox[3],img.shape[0],img.shape[1]),dtype=np.int)
    image=image[border[0]:border[1],border[2]:border[3],...]
    return image,border

def get_mask_BZ(img):
    if img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    threhold = np.mean(gray_img)/3-7
    _, mask = cv2.threshold(gray_img, max(0,threhold), 1, cv2.THRESH_BINARY)
    nn_mask = np.zeros((mask.shape[0]+2,mask.shape[1]+2),np.uint8)
    new_mask = (1-mask).astype(np.uint8)
    _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, (0,0), (0), cv2.FLOODFILL_MASK_ONLY)
    _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, (new_mask.shape[1]-1,new_mask.shape[0]-1), (0), cv2.FLOODFILL_MASK_ONLY)
    mask = mask + new_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,  20))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    return mask

def preprocess(img):
    mask, bbox, center, radius = get_mask(img)
    r_img = mask_image(img, mask)
    # r_img, r_border = remove_back_area(r_img, bbox=bbox)
    # mask, _ = remove_back_area(mask, border=r_border)
    #r_img, sup_border = supplemental_black_area(r_img)
    #mask, _ = supplemental_black_area(mask, border=sup_border)
    print(r_img.shape)
    print(mask.shape)
    return r_img, (mask * 255).astype(np.uint8)

'''
===== Degrade ====
'''


def DE_COLOR(img, brightness=0.3, contrast=0.4, saturation=0.4):
    """Randomly change the brightness, contrast and saturation of an image"""

    if brightness > 0:
        brightness_factor = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness - 0.1)  # brightness factor
        img = F.adjust_brightness(img, brightness_factor)
    if contrast > 0:
        contrast_factor = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)  # contrast factor
        img = F.adjust_contrast(img, contrast_factor)
    if saturation > 0:
        saturation_factor = random.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation)  # saturation factor
        img = F.adjust_saturation(img, saturation_factor)

    img = transform(img)
    img = img.numpy()

    color_params = {}
    color_params['brightness_factor'] = brightness_factor
    color_params['contrast_factor'] = contrast_factor
    color_params['saturation_factor'] = saturation_factor

    return img, color_params


def DE_HALO(img, h, w, brightness_factor, center=None, radius=None):
    '''
    Defined to simulate a 'ringlike' halo noise in fundus image
    :param weight_r/weight_g/weight_b: Designed to simulate 3 kinds of halo noise shown in Kaggle dataset.
    :param center_a/center_b:          Position of each circle which is simulated the ringlike shape
    :param dia_a/dia_b:                Size of each circle which is simulated the ringlike noise
    :param weight_hal0:                Weight of added halo noise color
    :param sigma:                      Filter size for final Gaussian filter
    '''

    weight_r = [251 / 255, 141 / 255, 177 / 255]
    weight_g = [249 / 255, 238 / 255, 195 / 255]
    weight_b = [246 / 255, 238 / 255, 147 / 255]
    # num
    if brightness_factor >= 0.2:
        num = random.randint(1, 2)
    else:
        num = random.randint(0, 2)
    w0_a = random.randint(w / 2 - int(w / 8), w / 2 + int(w / 8))
    h0_a = random.randint(h / 2 - int(h / 8), h / 2 + int(h / 8))
    center_a = [w0_a, h0_a]

    wei_dia_a = 0.75 + (1.0 - 0.75) * random.random()
    dia_a = min(h, w) * wei_dia_a
    Y_a, X_a = np.ogrid[:h, :w]
    dist_from_center_a = np.sqrt((X_a - center_a[0]) ** 2 + (Y_a - center_a[1]) ** 2)
    circle_a = dist_from_center_a <= (int(dia_a / 2))

    mask_a = np.zeros((h, w))
    mask_a[circle_a] = np.mean(img)  # np.multiply(A[0], (1 - t))

    center_b = center_a
    Y_b, X_b = np.ogrid[:h, :w]
    dist_from_center_b = np.sqrt((X_b - center_b[0]) ** 2 + (Y_b - center_b[1]) ** 2)

    dia_b_max = 2 * int(np.sqrt(
        max(center_a[0], h - center_a[0]) * max(center_a[0], h - center_a[0]) + max(center_a[1], h - center_a[1]) * max(
            center_a[1], w - center_a[1]))) / min(w, h)
    wei_dia_b = 1.0 + (dia_b_max - 1.0) * random.random()

    if num == 0:
        # if halo tend to be a white one, set the circle with a larger radius.
        dia_b = min(h, w) * wei_dia_b + abs(max(center_b[0] - w / 2, center_b[1] - h / 2) + max(w, h) * 2 / 3)
    else:
        dia_b = min(h, w) * wei_dia_b + abs(max(center_b[0] - w / 2, center_b[1] - h / 2) + max(w, h) / 2)

    circle_b = dist_from_center_b <= (int(dia_b / 2))

    mask_b = np.zeros((h, w))
    mask_b[circle_b] = np.mean(img)

    weight_hal0 = [0, 1, 1.5, 2, 2.5]
    delta_circle = np.abs(mask_a - mask_b) * weight_hal0[1]
    dia = max(center_a[0], h - center_a[0], center_a[1], h - center_a[1]) * 2
    gauss_rad = int(np.abs(dia - dia_a))
    sigma = 2 / 3 * gauss_rad

    if (gauss_rad % 2) == 0:
        gauss_rad = gauss_rad + 1
    delta_circle = cv2.GaussianBlur(delta_circle, (gauss_rad, gauss_rad), sigma)

    delta_circle = np.array([weight_r[num] * delta_circle, weight_g[num] * delta_circle, weight_b[num] * delta_circle])
    img = img + delta_circle

    img = np.maximum(img, 0)
    img = np.minimum(img, 1)

    halo_params = {}
    halo_params['num'] = num
    halo_params['center_a'] = center_a
    halo_params['dia_a'] = dia_a
    halo_params['center_b'] = center_b
    halo_params['dia_b'] = dia_b

    return img, halo_params


def DE_HOLE(img, h, w, region_mask, center=None, diameter=None):
    '''

    :param diameter_circle:     The size of the simulated artifacts caused by non-uniform lighting
    :param center:              Position
    :param brightness_factor:   Weight utilized to adapt the value of generated non-uniform lighting artifacts.
    :param sigma:               Filter size for final Gaussian filter

    :return:
    '''
    # if radius is None: # use the smallest distance between the center and image walls
    # diameter_circle = random.randint(int(0.3*w), int(0.5 * w))
    #  define the center based on the position of disc/cup
    diameter_circle = random.randint(int(0.4 * w), int(0.7 * w))

    center = [random.randint(w / 4, w * 3 / 4), random.randint(h * 3 / 8, h * 5 / 8)]
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    circle = dist_from_center <= (int(diameter_circle / 2))

    mask = np.zeros((h, w))
    mask[circle] = 1

    num_valid = np.sum(region_mask)
    aver_color = np.sum(img) / (3 * num_valid)
    if aver_color > 0.25:
        brightness = random.uniform(-0.26, -0.262)
        brightness_factor = random.uniform((brightness - 0.06 * aver_color), brightness - 0.05 * aver_color)
    else:
        brightness = 0
        brightness_factor = 0
    # print( (aver_color,brightness,brightness_factor))
    mask = mask * brightness_factor

    rad_w = random.randint(int(diameter_circle * 0.55), int(diameter_circle * 0.75))
    rad_h = random.randint(int(diameter_circle * 0.55), int(diameter_circle * 0.75))
    sigma = 2 / 3 * max(rad_h, rad_w) * 1.2

    if (rad_w % 2) == 0: rad_w = rad_w + 1
    if (rad_h % 2) == 0: rad_h = rad_h + 1

    mask = cv2.GaussianBlur(mask, (rad_w, rad_h), sigma)
    mask = np.array([mask, mask, mask])
    img = img + mask
    img = np.maximum(img, 0)
    img = np.minimum(img, 1)

    hole_params = {}
    hole_params['center'] = center
    hole_params['diameter_circle'] = diameter_circle
    hole_params['brightness_factor'] = brightness_factor
    hole_params['rad_w'] = rad_w
    hole_params['rad_h'] = rad_h
    hole_params['sigma'] = sigma

    return img, hole_params


def DE_ILLUMINATION(img, region_mask, h=1024, w=1024):
    img, color_params = DE_COLOR(img)
    img, halo_params = DE_HALO(img, h, w, color_params['brightness_factor'])
    img, hole_params = DE_HOLE(img, h, w, region_mask)

    illum_params = {}
    illum_params['color'] = color_params
    illum_params['halo'] = halo_params
    illum_params['hole'] = hole_params

    return img, illum_params


def DE_SPOT(img, h, w, center=None, radius=None):
    '''
    :param s_num:  The number of the generated artifacts spot on the fundus image
    :param radius: Define the size of each spot
    :param center: Position of each spot on the fundus image
    :param K:      Weight of original fundus image value
    :param beta:   Weight of generated artifacts(spots) mask value (The color is adapted based on the size(radius) of each spot)
    :param sigma:  Filter size for final Gaussian filter

    '''
    spot_params = []
    s_num = random.randint(5, 10)
    mask0 = np.zeros((h, w))
    for i in range(s_num):
        # if radius is None: # use the smallest distance between the center and image walls
        # radius = min(center[0], center[1], w-center[0], h-center[1])
        radius = random.randint(math.ceil(0.01 * h), int(0.05 * h))

        # if center is None: # in the middle of the image
        center = [random.randint(radius + 1, w - radius - 1), random.randint(radius + 1, h - radius - 1)]
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        circle = dist_from_center <= (int(radius / 2))

        k = (14 / 25) + (1.0 - radius / 25)
        beta = 0.5 + (1.5 - 0.5) * (radius / 25)
        A = k * np.ones((3, 1))
        d = 0.3 * (radius / 25)
        t = math.exp(-beta * d)

        mask = np.zeros((h, w))
        mask[circle] = np.multiply(A[0], (1 - t))
        mask0 = mask0 + mask
        mask0[mask0 != 0] = 1

        sigma = (5 + (20 - 0) * (radius / 25)) * 2
        rad_w = random.randint(int(sigma / 5), int(sigma / 4))
        rad_h = random.randint(int(sigma / 5), int(sigma / 4))
        if (rad_w % 2) == 0: rad_w = rad_w + 1
        if (rad_h % 2) == 0: rad_h = rad_h + 1

        mask = cv2.GaussianBlur(mask, (rad_w, rad_h), sigma)
        mask = np.array([mask, mask, mask])
        img = img + mask
        img = np.maximum(img, 0)
        img = np.minimum(img, 1)

        spot_dict = {}
        spot_dict['radius'] = radius
        spot_dict['center'] = center
        spot_dict['rad_w'] = rad_w
        spot_dict['rad_h'] = rad_h
        spot_dict['sigma'] = sigma
        spot_params.append(spot_dict)

    return img, spot_params


def DE_BLUR(img, h, w, center=None, radius=None):
    '''

    :param sigma: Filter size for Gaussian filter

    '''
    img = (np.transpose(img, (1, 2, 0)))
    sigma = 5 + (15 - 5) * random.random()
    rad_w = random.randint(int(sigma / 3), int(sigma / 2))
    rad_h = random.randint(int(sigma / 3), int(sigma / 2))
    if (rad_w % 2) == 0: rad_w = rad_w + 1
    if (rad_h % 2) == 0: rad_h = rad_h + 1

    img = cv2.GaussianBlur(img, (rad_w, rad_h), sigma)
    img = (np.transpose(img, (2, 0, 1)))

    img = np.maximum(img, 0)
    img = np.minimum(img, 1)

    blur_params = {}
    blur_params['sigma'] = sigma
    blur_params['rad_w'] = rad_w
    blur_params['rad_h'] = rad_h

    return img, blur_params


def DE_process(img, mask, h, w, de_type):
    params = {}
    if de_type == '001':
        img, illum_params = DE_ILLUMINATION(img, mask, h, w)
        params['illumination'] = illum_params
    elif de_type == '010':
        img = transform(img)
        img = img.numpy()
        img, spot_params = DE_SPOT(img, h, w)
        params['spots'] = spot_params
    elif de_type == '011':
        img, illum_params = DE_ILLUMINATION(img, mask, h, w)
        params['illumination'] = illum_params
        img, spot_params = DE_SPOT(img, h, w)
        params['spots'] = spot_params
    elif de_type == '100':
        img = transform(img)
        img = img.numpy()
        img, blur_params = DE_BLUR(img, h, w)
        params['blur'] = blur_params
    elif de_type == '101':
        img, illum_params = DE_ILLUMINATION(img, mask, h, w)
        params['illumination'] = illum_params
        img, blur_params = DE_BLUR(img, h, w)
        params['blur'] = blur_params
    elif de_type == '110':
        img = transform(img)
        img = img.numpy()
        img, spot_params = DE_SPOT(img, h, w)
        params['spots'] = spot_params
        img, blur_params = DE_BLUR(img, h, w)
        params['blur'] = blur_params
    elif de_type == '111':
        img, illum_params = DE_ILLUMINATION(img, mask, h, w)
        params['illumination'] = illum_params
        img, spot_params = DE_SPOT(img, h, w)
        params['spots'] = spot_params
        img, blur_params = DE_BLUR(img, h, w)
        params['blur'] = blur_params
    else:
        raise ValueError('Wrong type')

    img = (np.transpose(img*mask, (1,2,0))*255).astype(np.uint8)

    return img, params

def get_transform(resize_or_crop,loadSizeX,loadSizeY,fineSize):
    transform_list = []
    if resize_or_crop == 'resize_and_crop':
        osize = [loadSizeX, loadSizeY]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(fineSize))
    elif resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(fineSize))
    elif resize_or_crop =='scale':
        osize = [loadSizeX,loadSizeY]

    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)

transform = get_transform('scale',loadSizeX =1024, loadSizeY=1024, fineSize=1024)

def run(image_path, output_dir='./', gt_dir='./', mask_dir='./'):
    print(image_path)

    image = Image.open(image_path).convert('RGB')
    h, w = image.size
    _, image_id = os.path.split(image_path)
    image_id = image_id.split('.')[0]
    dsize = (1024, 1024)
    image, mask = preprocess(np.asarray(image).copy())
    image = cv2.resize(image, dsize)
    mask = cv2.resize(mask, dsize)
    image = Image.fromarray(image, mode='RGB')
    mask = Image.fromarray(mask, mode='L')
    mask_path = os.path.join(mask_dir, image_id + '.jpeg')
    mask.save(mask_path)
    mask = np.expand_dims(mask, axis=2)
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    gt_path = os.path.join(gt_dir, image_id + '.jpeg')
    image.save(gt_path)
    params = ['001', '100', '010', '110', '101', '011', '111']
    for param in params:
        output_image, params = DE_process(image, mask, 1024, 1024, param)
        output_path = os.path.join(output_dir, image_id + '_{}.jpeg'.format(param))
        imsave(output_path, output_image)



if __name__ == '__main__':
    from multiprocessing import Pool
    from tqdm import tqdm
    from glob import glob
    pool = Pool(processes=16)
    results = []
    if not os.path.exists(args.mask_dir):
        os.makedirs(args.mask_dir)
    if not os.path.exists(args.gt_dir):
        os.makedirs(args.gt_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(args.test_dir)
    print(os.listdir(args.test_dir))
    image_list = sorted(glob(os.path.join(args.test_dir, '*')))
    print(image_list)
    for image_path in tqdm(image_list):
        results.append(pool.apply_async(run, (image_path, args.output_dir, args.gt_dir, args.mask_dir)))
    pool.close()
    for res in results:
        res.get()






