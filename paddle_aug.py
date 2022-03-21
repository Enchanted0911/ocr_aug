import random
import numpy as np
import cv2
import os

from text_image_aug import tia_perspective, tia_stretch, tia_distort


class RecAug(object):
    def __init__(self, use_tia=True, aug_prob=1, num=9):
        self.use_tia = use_tia
        self.aug_prob = aug_prob
        self.num = num

    def __call__(self, data):
        img = data['image']
        return warp(img, self.num)


def flag():
    """
    flag
    """
    return 1 if random.random() > 0.5000001 else -1


def cvt_color(img):
    """
    cvtColor
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    delta = 0.001 * random.random() * flag()
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


def blur(img):
    """
    blur
    """
    h, w, _ = img.shape
    if h > 10 and w > 10:
        return cv2.GaussianBlur(img, (5, 5), 1)
    else:
        return img


def jitter(img):
    """
    jitter
    """
    w, h, _ = img.shape
    if h > 10 and w > 10:
        thres = min(w, h)
        s = int(random.random() * thres * 0.01)
        src_img = img.copy()
        for i in range(s):
            img[i:, i:, :] = src_img[:w - i, :h - i, :]
        return img
    else:
        return img


def add_gasuss_noise(image, mean=0, var=0.1):
    """
    Gasuss noise
    """

    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out


def get_crop(image):
    """
    random crop
    """
    h, w, _ = image.shape
    top_min = 1
    top_max = 8
    top_crop = int(random.randint(top_min, top_max))
    top_crop = min(top_crop, h - 1)
    crop_img = image.copy()
    ratio = random.randint(0, 1)
    if ratio:
        crop_img = crop_img[top_crop:h, :, :]
    else:
        crop_img = crop_img[0:h - top_crop, :, :]
    return crop_img


def warp(img, num=9):
    """
    warp
    """
    h, w, _ = img.shape
    new_img = img
    img_height, img_width = img.shape[0:2]
    image_list = []
    for selected_aug in combination_generator(num):
        if 0 in selected_aug and img_height >= 20 and img_width >= 20:
            new_img = tia_distort(new_img, random.randint(3, 6))

        if 1 in selected_aug and img_height >= 20 and img_width >= 20:
            new_img = tia_stretch(new_img, random.randint(3, 6))

        if 2 in selected_aug:
            new_img = tia_perspective(new_img)

        if 3 in selected_aug and img_height >= 20 and img_width >= 20:
            new_img = get_crop(new_img)

        if 4 in selected_aug:
            new_img = blur(new_img)

        if 5 in selected_aug:
            new_img = cvt_color(new_img)

        # if 61 in selected_aug:
        #     new_img = jitter(new_img)
        #
        # if 71 in selected_aug:
        #     new_img = add_gasuss_noise(new_img)

        if 6 in selected_aug:
            new_img = 255 - new_img

        image_list.append(new_img)
        new_img = img
    return image_list


def combination_generator(num=9):
    all_combination_list = set([])
    base_list = []
    for i in range(num):
        base_list.append(i)

    no_name_function(all_combination_list, base_list)

    return all_combination_list


def no_name_function(one_list, base_list):
    one_list.add(tuple(base_list))
    if len(base_list) < 2:
        return

    for i in range(len(base_list) - 1, -1, -1):
        temp = base_list[:]
        del temp[i]
        no_name_function(one_list, temp)


def generate_aug_img(img_path, generate_path):
    if not os.path.exists(generate_path):
        os.makedirs(generate_path)
    img = cv2.imread(img_path)
    img_data = {'image': img}
    aug = RecAug(True, 1, 7)
    image_list = aug(img_data)
    cnt = 1
    for i in image_list:
        cv2.imwrite(generate_path + str(cnt) + '.jpg', i)
        cnt += 1


if __name__ in '__main__':
    l_boundary = 102
    r_boundary = 110
    for j in range(l_boundary, r_boundary):
        i_num = j
        i_path = "D:\\wjs\\aug_images\\0" + str(i_num) + ".png"
        g_path = "D:\\wjs\\aug_generate\\" + str(i_num - 1) + "\\"
        generate_aug_img(i_path, g_path)
