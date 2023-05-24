import os
import math
import argparse

from tqdm import tqdm
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

parser = argparse.ArgumentParser()
parser.add_argument("--gt_path", type=str, required=True)
parser.add_argument("--inf_path", type=str, required=True)
args = parser.parse_args()

gt_path = args.gt_path
inf_path = args.inf_path


def psnr(target, ref):
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref, dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten("C")
    rmse = math.sqrt(np.mean(diff**2.0))
    eps = np.finfo(np.float64).eps
    if rmse == 0:
        rmse = eps
    return 20 * math.log10(255.0 / rmse)


def myssim(target, ref):
    return ssim(ref, target, multichannel=True, channel_axis=-1)


if __name__ == "__main__":
    # gt_list = os.listdir(path=gt_path)
    inf_list = os.listdir(path=inf_path)
    psnr_list = []
    ssim_list = []
    for inf in tqdm(inf_list):
        if not os.path.isdir(os.path.join(inf_path, inf)):
            continue
        img_gt = Image.open(os.path.join(gt_path, "{}.png".format(inf))).convert("RGB")
        inf_img_list = os.listdir(os.path.join(inf_path, inf))
        for inf_img in inf_img_list:
            img_inf = Image.open(os.path.join(inf_path, inf, inf_img)).convert("RGB")
            img_inf = img_inf.crop((2, 2, 9612, 1082))
            frame = None
            sub_width = 1920
            for i in range(5):
                # 计算子图像的左上角和右下角坐标
                left = i * sub_width + i * 2
                top = 0
                right = left + sub_width
                bottom = 1080
                # print((left, top, right, bottom))
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
            psnr_list.append(psnr(img_gt, mean_frame))
            ssim_list.append(myssim(np.array(img_gt), mean_frame))
        # print("PSNR: ", np.mean(psnr_list))
        # print("SSIM: ", np.mean(ssim_list))
    print("PSNR: ", np.mean(psnr_list))
    print("SSIM: ", np.mean(ssim_list))
# img_gt = Image.open("ground_truth.png").convert("RGB")
# img_recovered = Image.open("recovered.png").convert("RGB")
