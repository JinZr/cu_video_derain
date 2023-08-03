import os
import math
import argparse

from tqdm import tqdm
from PIL import Image
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--out_path", type=str, required=True)
parser.add_argument("--inf_path", type=str, required=True)
args = parser.parse_args()

out_path = args.out_path
inf_path = args.inf_path


if __name__ == "__main__":
    # gt_list = os.listdir(path=gt_path)
    inf_list = os.listdir(path=inf_path)
    psnr_list = []
    ssim_list = []
    for inf in tqdm(inf_list):
        if not os.path.isdir(os.path.join(inf_path, inf)):
            continue
        inf_img_list = os.listdir(os.path.join(inf_path, inf))
        for inf_img in inf_img_list:
            img_inf = Image.open(os.path.join(inf_path, inf, inf_img)).convert("RGB")
            frame = None
            sub_width = 1920
            for i in range(5):
                # 计算子图像的左上角和右下角坐标
                left = i * sub_width
                top = 0
                right = left + sub_width
                bottom = 1080

                # 分离子图像
                sub_image = img_inf.crop((left, top, right, bottom))
                # print(np.array(sub_image).shape)
                # print(frame.shape if frame is not None else None)
                if frame is None:
                    frame = np.stack([np.array(sub_image)], axis=0)
                else:
                    frame = np.concatenate(
                        [frame, np.stack([np.array(sub_image)], axis=0)], axis=0
                    )

                # 保存子图像
            mean_frame = np.min(frame, axis=0)
            Image.fromarray(mean_frame).save(os.path.join(out_path, inf, inf_img))
