import os
import time

import cv2
import torch
import argparse
import torchvision
import numpy as np
import scipy.io as scio
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import utils
import models
from lpips.lpips import *

parser = argparse.ArgumentParser(description="Args of this repo.")
parser.add_argument("--rate", default=0.04, type=float)
parser.add_argument("--device", default="0")
opt = parser.parse_args()
opt.device = "cuda:" + opt.device


def evaluate():
    print("Start evaluate...")
    config = utils.GetConfig(ratio=opt.rate, device=opt.device)
    net = models.HybridNet(config).to(config.device).eval()

    if os.path.exists(config.model):
        if torch.cuda.is_available():
            trained_model = torch.load(config.model, map_location=config.device)
        else:
            trained_model = torch.load(config.model, map_location="cpu")

        net.load_state_dict(trained_model)
        print("Trained model loaded.")
    else:
        raise FileNotFoundError("Missing trained models.")

    res(config, net, save_img=False)


def res(config, net, save_img):
    tensor2image = torchvision.transforms.ToPILImage()
    save_img = save_img
    batch_size = 1

    net = net.eval()
    file_no = [
       11
    ]

    folder_name = [
        "Set11_GREY",
    ]

    for idx, item in enumerate(folder_name):
        p_total = 0
        s_total = 0
        mse_total = 0
        path = os.path.join("G:/dataset/PNG/Grey", item)
        # path = os.path.join("/home/wcr/WXY/dataset/PNG/Grey", item)
        print("*", ("  test dataset: " + path + ", device: " + str(config.device) + "  ").center(120, "="), "*")
        with torch.no_grad():
            for root, dir, files in os.walk(path):
                i = 0
                for file in files:
                    if file[-4:] == ".png" or file[-4:] == ".tif" or file[-4:] == ".bmp":
                        i = i + 1
                        name = os.path.join(path, file)
                        img = cv2.imread(name)
                        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                        img_rec_yuv = img_yuv.copy()
                        img_y = img_yuv[:, :, 0]/255.
                        x = img_y
                        x = torch.from_numpy(np.array(x)).to(config.device)
                        x = x.float()
                        ori_x = x

                        h = x.size()[0]
                        h_lack = 0
                        w = x.size()[1]
                        w_lack = 0

                        if h % config.block_size != 0:
                            h_lack = config.block_size - h % config.block_size
                            temp_h = torch.zeros(h_lack, w).to(config.device)
                            h = h + h_lack
                            x = torch.cat((x, temp_h), 0)

                        if w % config.block_size != 0:
                            w_lack = config.block_size - w % config.block_size
                            temp_w = torch.zeros(h, w_lack).to(config.device)
                            w = w + w_lack
                            x = torch.cat((x, temp_w), 1)

                        x = torch.unsqueeze(x, 0)
                        x = torch.unsqueeze(x, 0)

                        idx_h = range(0, h, config.block_size)
                        idx_w = range(0, w, config.block_size)
                        num_patches = h * w // (config.block_size ** 2)

                        temp = torch.zeros(num_patches, batch_size, config.channel, config.block_size, config.block_size)
                        count = 0
                        for a in idx_h:
                            for b in idx_w:

                                ori = x[:, :, a:a + config.block_size, b:b + config.block_size].to(config.device)
                                output = net(ori)
                                temp[count, :, :, :, :, ] = output
                                count = count + 1

                        y = torch.zeros(batch_size, config.channel, h, w)
                        count = 0
                        for a in idx_h:
                            for b in idx_w:
                                y[:, :, a:a + config.block_size, b:b + config.block_size] = temp[count, :, :, :, :]
                                count = count + 1

                        recon_x = y[:, :, 0:h - h_lack, 0:w - w_lack]

                        recon_x = torch.squeeze(recon_x).to("cpu")
                        ori_x = ori_x.to("cpu")

                        psnr = PSNR(recon_x.numpy(), ori_x.numpy(), data_range=1)
                        ssim = SSIM(recon_x.numpy(), ori_x.numpy(), data_range=1)

                        p_total = p_total + psnr
                        s_total = s_total + ssim

                        if save_img:
                            img_path = "./recon_img_Y/{}/{}/".format(item, int(config.ratio * 100))
                            if not os.path.isdir("./recon_img_Y/{}/".format(item)):
                                os.mkdir("./recon_img_Y/{}/".format(item))
                            if not os.path.isdir(img_path):
                                os.mkdir(img_path)
                                print("\rMkdir {}".format(img_path))

                            img_rec_yuv[:, :, 0] = recon_x * 255
                            im_rec_rgb = cv2.cvtColor(img_rec_yuv, cv2.COLOR_YCrCb2BGR)
                            im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

                            fileName = file[0:(len(file)-4)]
                            # im_rec_rgb.save(img_path + "{}_{}_{}.png".format(fileName, round(psnr,2), round(ssim,4)))
                            cv2.imwrite(f"{img_path}/{fileName}_{round(psnr,4)}_{round(ssim,4)}.png", (im_rec_rgb))
                print("=> All the {:2} images done!, your AVG PSNR: {:5.4f}, AVG SSIM: {:5.4f}"
                  .format(file_no[idx], p_total / file_no[idx], s_total / file_no[idx]))


if __name__ == "__main__":
    evaluate()
