import os
import time
import torch
import argparse
import torchvision
import numpy as np
import scipy.io as scio
import pandas as pd
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import utils
import models

parser = argparse.ArgumentParser(description="Args of this repo.")
parser.add_argument("--rate", default=0.01, type=float)
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

    res(config, net, save_img = False)


def res(config, net, save_img):
    tensor2image = torchvision.transforms.ToPILImage()
    save_img = save_img
    batch_size = 1

    net = net.eval()
    file_no = [
        100,
    ]

    folder_name = [
        "General100",
    ]
    time_sum = 0
    time_All = []
    for idx, item in enumerate(folder_name):
        p_total = 0
        s_total = 0
        path = os.path.join(config.test_path, item)
        print("*", ("  test dataset: " + path + ", device: " + str(config.device) + "  ").center(120, "="), "*")
        with torch.no_grad():
            for root, dir, files in os.walk(path):
                i = 0
                for file in files:
                    if file[-4:] == ".mat":
                        i = i + 1
                        name = os.path.join(path, file)
                        x = scio.loadmat(name)['data']
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

                        x = torch.cat(torch.split(x, split_size_or_sections=config.block_size, dim=3),dim=0)
                        x = torch.cat(torch.split(x, split_size_or_sections=config.block_size, dim=2),dim=0)
                        idx_h = range(0, h, config.block_size)
                        idx_w = range(0, w, config.block_size)
                        num_patches = h * w // (config.block_size ** 2)

                        temp = torch.zeros(num_patches, batch_size, config.channel, config.block_size, config.block_size)
                        count = 0
                        for a in idx_h:
                            for b in idx_w:

                                ori = x[:, :, a:a + config.block_size, b:b + config.block_size].to(config.device)
                                t1 = time.time()
                                output = net(ori)
                                t2 = time.time()
                                time_sum = time_sum + t2-t1
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
                            img_path = "./recon_img/{}/{}/".format(item, int(config.ratio * 100))
                            if not os.path.isdir("./recon_img/{}/".format(item)):
                                os.mkdir("./recon_img/{}/".format(item))
                            if not os.path.isdir(img_path):
                                os.mkdir(img_path)
                                print("\rMkdir {}".format(img_path))
                            recon_x = tensor2image(recon_x)
                            fileName = file[0:(len(file)-4)]
                            recon_x.save(img_path + "{}_{}_{}.png".format(fileName, round(psnr,2), round(ssim,4)))

                print("=> All the {:2} images done!, your AVG PSNR: {:5.2f}, AVG SSIM: {:5.4f}"
                  .format(file_no[idx], p_total / file_no[idx], s_total / file_no[idx]))

if __name__ == "__main__":
    evaluate()
