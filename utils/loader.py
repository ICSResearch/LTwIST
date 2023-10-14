import os
import cv2
import copy
import time
import torch
import random
import torchvision
import numpy as np
from PIL import Image
import torch.utils.data as data

import utils

class DataAugment:
    def __init__(self, debug=False):
        self.debug = debug

    def basic_matrix(self, translation):
        return np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    def adjust_transform_for_image(self, img, trans_matrix):
        transform_matrix = copy.deepcopy(trans_matrix)  #深拷贝，会拷贝对象及其子对象，哪怕以后对其有改动，也不会影响其第一次的拷贝
        # height, width, channels = img.shape
        # print(img.shape)
        height, width = img.shape
        transform_matrix[0:2, 2] *= [width, height]

        # print("loader:trans_matrix:{}".format(trans_matrix.shape))
        # print("loader:transform_matrix1:{}".format(transform_matrix.shape))

        center = np.array((0.5 * width, 0.5 * height))
        transform_matrix = np.linalg.multi_dot([self.basic_matrix(center), transform_matrix, self.basic_matrix(-center)])

        # print("loader:center:{}".format(center))
        # print("loader:basic_matrix:{}".format(self.basic_matrix(center).shape))
        # print("loader:transform_matrix2:{}".format(transform_matrix.shape))

        return transform_matrix


    def apply(self, img, trans_matrix):
        tmp_matrix = self.adjust_transform_for_image(img, trans_matrix)

        # print("trans_matrix:{}".format(trans_matrix.shape))
        # print("tmp_matrix:{}".format(tmp_matrix.shape))

        # cv2.warpAffine():仿射变函数，实现图像旋转
        out_img = cv2.warpAffine(img, tmp_matrix[:2, :], dsize=(img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT, borderValue=0,)
        # print("out_img:{}".format(out_img.shape))

        return out_img

    def random_vector(self, min, max):
        min = np.array(min)
        max = np.array(max)
        return np.random.uniform(min, max)

    # 图像旋转
    def random_rotate(self, img, factor):
        angle = np.random.uniform(factor[0], factor[1])
        rotate_matrix = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        out_img = self.apply(img, rotate_matrix)
        return rotate_matrix, out_img

    # 图像扩大
    def random_scale(self, img, min_translation, max_translation):
        factor = self.random_vector(min_translation, max_translation)
        scale_matrix = np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]])
        out_img = self.apply(img, scale_matrix)
        return scale_matrix, out_img


class TrainDataPackage:
    def __init__(self, root="./dataset/train/", transform=None, packaged=True):  # todo ./dataset
        self.training_file = "train_coco800.pt"
        self.aug = DataAugment(debug=True)
        self.packaged = packaged
        self.root = root
        self.num = 200

        # torchvision.transforms.Compose():串联多个图片变换的操作
        self.transform = transform or torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomVerticalFlip(),        # 随机垂直翻转
            torchvision.transforms.RandomHorizontalFlip()])      # 随机水平翻转
            # torchvision.transforms.Grayscale(num_output_channels=1)])   #将图像转为灰度图

        if not (os.path.exists(os.path.join(self.root, self.training_file))):
            print("No packaged dataset file (*.pt) in dataset/, Now generating...")
            self.generate()

        if packaged:
            self.train_data = torch.load(os.path.join(self.root, self.training_file))

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        img = self.train_data[index]
        return img

    def generate(self):
        paths = [
            os.path.join(self.root, "coco/")
            # os.path.join(self.root, "BSD500/train/train"),
            # os.path.join(self.root, "BSD500/test/"),
            # os.path.join(self.root, "BSD500/val/"),
        ]
        patches_list = []

        start = time.time()
        for path in paths:
            for roots, dirs, files in os.walk(path):
                if roots == path:
                    print("Image number: {}".format(files.__len__()))
                    for file in files:
                        if file[-4:] == ".jpg" or file[-4:] == ".png":
                            temp = os.path.join(path, file)
                            print("=> Processing " + temp)
                            # image = Image.open(temp).convert('L')
                            # patches = self.random_patch(image, self.num)
                            # patches_list.extend(patches)
                            # 用亮度分量打包处理
                            Img = cv2.imread(temp)
                            image_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
                            image_y = image_yuv[:,:,0]
                            # image_y = Image.fromarray(np.uint8(image_y))
                            patches = self.random_patch(image_y, self.num)
                            patches_list.extend(patches)
        print("Total patches: {}".format(patches_list.__len__()))
        print("Now Packaging...")
        with open(os.path.join(self.root, self.training_file), 'wb') as f:
            torch.save(patches_list, f)
        end = time.time()
        print("Successfully packaged!, used time: {:.3f}".format(end - start))

    def random_patch(self, image, num):
        size = 96
        image = np.array(image, dtype=np.float32) / 255     # 图像像素归一化
        h, w = image.shape[0], image.shape[1]
        patches = []
        for n in range(num):
            max_h = random.randint(0, h - size)     # 随机生成0到h-size的一个整数
            max_w = random.randint(0, w - size)
            patch = image[max_h:max_h + size, max_w:max_w + size]

            if 0 <= n <= 50:
                _, patch = self.aug.random_rotate(patch, (0., 1.))
            if 50 <= n <= 100:
                _, patch = self.aug.random_scale(patch, (1., 1.), (2., 2.))

            patch = self.transform(patch)
            patches.append(patch)
        return patches

def train_loader(batch_size):
    train_dataset = TrainDataPackage()
    dst = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True,
                                      shuffle=True, pin_memory=True, num_workers=8)
    return dst


if __name__ == '__main__':
    my_config = utils.GetConfig()
    # TrainDataPackage()
    print("Now Loading train data...")
    dst = train_loader(my_config.batch_size)
    print("Train data loaded, length: {}".format(dst.__len__()))
    for x in dst:
        print(x.size())
