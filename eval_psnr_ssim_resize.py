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
        img_gt = (
            Image.open(os.path.join(gt_path, "{}.png".format(inf)))
            .convert("RGB")
            .resize((910, 512))
        )
        inf_img_list = os.listdir(os.path.join(inf_path, inf))
        for inf_img in inf_img_list:
            img_inf = Image.open(os.path.join(inf_path, inf, inf_img)).convert("RGB")
            psnr_list.append(psnr(img_gt, img_inf))
            ssim_list.append(myssim(np.array(img_gt), np.array(img_inf)))
    print("PSNR: ", np.mean(psnr_list))
    print("SSIM: ", np.mean(ssim_list))
# img_gt = Image.open("ground_truth.png").convert("RGB")
# img_recovered = Image.open("recovered.png").convert("RGB")
