import random
import numpy as np
import cv2
import os

from text_image_aug import tia_perspective, tia_stretch, tia_distort
from text_image_aug.file_util import write_str_list_to_txt


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
    delta = 1 * random.random() * flag()
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


def blur(img, x):
    """
    blur
    """
    h, w, _ = img.shape
    if h > 10 and w > 10:
        return cv2.GaussianBlur(img, (x, x), 1)
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
        # if 0 in selected_aug and img_height >= 20 and img_width >= 20:
        #     new_img = tia_distort(new_img, random.randint(3, 6))
        if 0 in selected_aug:
            new_img = add_gasuss_noise(new_img)

        # if 1 in selected_aug and img_height >= 20 and img_width >= 20:
        # new_img = tia_stretch(new_img, random.randint(3, 6))
        if 1 in selected_aug:
            # new_img = jitter(new_img)
            new_img = blur(new_img, 9)
        if 2 in selected_aug:
            # new_img = tia_perspective(new_img)
            new_img = blur(new_img, 7)
        if 3 in selected_aug and img_height >= 20 and img_width >= 20:
            new_img = get_crop(new_img)

        if 4 in selected_aug:
            new_img = blur(new_img, 5)

        if 5 in selected_aug:
            new_img = cvt_color(new_img)

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


def generate_aug_char(img_path, img_dir):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_path = img_path.replace("\\", "/")
    img_name = img_path[img_path.rfind("/") + 1: -4]

    img = cv2.imread(img_path)
    img_data = {'image': img}
    aug = RecAug(True, 1, 7)
    image_list = aug(img_data)
    cnt = 1
    for i in image_list:
        res_path = os.path.join(img_dir, img_name + "_aug" + str(cnt) + ".jpg")
        cv2.imwrite(res_path, i)
        cnt += 1


def auto_aug():
    l_boundary = 367
    r_boundary = 381
    for j in range(l_boundary, r_boundary):
        i_num = j
        i_path = "D:\\wjs\\aug_images\\0" + str(i_num) + ".png"
        g_path = "D:\\wjs\\aug_generate\\" + str(i_num - 1) + "\\"
        generate_aug_img(i_path, g_path)


def blur_all(rec_txt, parent_dir, blur_dir):
    with open(rec_txt, "r", encoding='utf8') as f:
        rec_list = f.readlines()
    rec_list = list(filter(lambda x: "aug_generate" not in x, rec_list))
    blur_txt = []

    if not os.path.exists(parent_dir + blur_dir):
        os.makedirs(parent_dir + blur_dir)

    for r in rec_list:
        try:
            pic_path = parent_dir + r.split("\t")[0]
            pic_name = pic_path[pic_path.rfind("/") + 1: -4]
            txt = r.split("\t")[1].replace("\n", "")
            img = cv2.imread(pic_path)
            blur_img = blur(img, 9)
            blur_path = parent_dir + blur_dir + pic_name + "_blur.jpg"
            blur_label = blur_dir + pic_name + "_blur.jpg" + "\t" + txt
            blur_txt.append(blur_label)
            cv2.imwrite(blur_path, blur_img)
        except Exception as e:
            print("err---------------->", e, r)
    write_str_list_to_txt(blur_txt, parent_dir + "blur9.txt")


def cvt_color_all(label_txt, parent_dir, cvt_dir):
    with open(label_txt, "r", encoding='utf8') as f:
        det_list = f.readlines()
    det_list = list(filter(lambda x: "aug_generate" not in x, det_list))
    cvt_txt = []

    if not os.path.exists(parent_dir + cvt_dir):
        os.makedirs(parent_dir + cvt_dir)

    for d in det_list:
        try:
            pic_path = parent_dir + d.split("\t")[0]
            pic_name = pic_path[pic_path.rfind("/") + 1: -4]
            txt = d.split("\t")[1].replace("\n", "")
            img = cv2.imread(pic_path)
            cvt_img = cvt_color(img)
            cvt_path = parent_dir + cvt_dir + pic_name + "_cvt.jpg"
            cvt_label = cvt_dir + pic_name + "_cvt.jpg" + "\t" + txt
            cvt_txt.append(cvt_label)
            cv2.imwrite(cvt_path, cvt_img)
        except Exception as e:
            print("err---------------->", e, d)
    write_str_list_to_txt(cvt_txt, parent_dir + "cvt.txt")


def aug_char(dir_path):
    char_dir_list = os.listdir(dir_path)
    for char_dir in char_dir_list:
        if "_aug" in char_dir:
            continue
        print("start char dir --------", char_dir)
        aug_dir = os.path.join(dir_path, char_dir + "_aug")
        if not os.path.exists(aug_dir):
            os.makedirs(aug_dir)
        char_dir_path = os.path.join(dir_path, char_dir)
        pic_list = os.listdir(char_dir_path)
        for pic_name in pic_list:
            print("start pic ------", pic_name)
            pic_path = os.path.join(char_dir_path, pic_name)
            generate_aug_char(pic_path, aug_dir)


if __name__ in '__main__':
    # d_path = r"D:\aug_char"
    # aug_char(d_path)


    dirx = "D:/wjs/ocr_temp_train/merge_data/rec_temp_train/"
    # cvt_color_all(r"D:/wjs/ocr_temp_train/merge_data/engList.txt", dirx, "cvt/")
    blur_all(r"D:/wjs/ocr_temp_train/merge_data/rec_temp_train/engList.txt", dirx, "blur9/")
    # auto_aug()
    # dirx = "D:/wjs/ocr_train_vin/merge_data/"
    # cvt_color_all(r"D:/wjs/ocr_train_vin/merge_data/new_Label.txt", dirx, "cvt_vin/")
    # dirx1 = "D:/wjs/ocr_train_nameplate/merge_data/"
    # cvt_color_all(r"D:/wjs/ocr_train_nameplate/merge_data/new_Label.txt", dirx1, "cvt_nameplate/")
