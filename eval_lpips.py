import glob
import os

# create default evaluator for doctests
import lpips
import torch
import torchvision.transforms as transforms
from PIL import Image

loss_fn_alex = lpips.LPIPS(net="alex")  # best forward scores


def eval_step(engine, batch):
    return batch


path = "/Users/zengruijin/Downloads/Ours/mid_heavy"
gt_path = "/Users/zengruijin/Downloads/test_heavy/final-static-img"
imgs = glob.glob(f"{path}/*/*.png", recursive=True, root_dir=path)
dirs = os.listdir(path)
transform = transforms.Compose([transforms.PILToTensor()])


def crop_img(img_pth):
    img_inf = Image.open(img_pth).convert("RGB") / 255.0
    img_inf = img_inf.crop((2, 2, 9612, 1082))
    frame = None
    sub_width = 1920
    for i in range(5):
        # 计算子图像的左上角和右下角坐标
        left = i * sub_width + i * 2
        top = 0
        right = left + sub_width
        bottom = 1080
        sub_image = img_inf.crop((left, top, right, bottom))
        if i == 2:
            frame = sub_image
            return frame
    return frame


if __name__ == "__main__":
    import sys

    key = sys.argv[1]
    paths = {
        "Ours": "/Users/zengruijin/Downloads/Ours/mid_motion",
        "ECCV22": "/Users/zengruijin/Downloads/ECCV22/heavy_ours",
        # "nafnet": "/Users/zengruijin/Downloads/nafnet/CUHK_test_motion/",
        "S2VD": "/Users/zengruijin/Downloads/S2VD/S2VD-ours_final_heavy",
        "IDT": "/Users/zengruijin/Downloads/IDT/heavy",
        # "ESTI_orig": "/Users/zengruijin/Downloads/ESTINet_orig/motion",
        # "ESTI": "/Users/zengruijin/Downloads/ESTINet_Retrained/min_motion_ours",
    }
    for key in paths.keys():
        path = paths[key]
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        scores = []
        for dir in dirs:
            score = 0
            imgs = glob.glob(f"{path}/{dir}/*.png", recursive=True, root_dir=path)
            for img in imgs:
                # img_name = img.split("/")[-1]
                gt_img = f"{gt_path}/{dir}.png"
                gt_img = (
                    transform(Image.open(gt_img).convert("RGB")).unsqueeze(0).to(device)
                ) / 255.0

                if key in ["Ours"]:
                    img = (
                        transform(Image.open(img).convert("RGB"))
                        .unsqueeze(0)
                        .to(device)
                    ) / 255.0
                else:
                    img = (
                        transform(Image.open(img).convert("RGB"))
                        .unsqueeze(0)
                        .to(device)
                    ) / 255.0

                score = loss_fn_alex(img, gt_img)
                scores.append(score)
        import numpy as np

        print(f"{key} niqe", np.mean(scores))
        print(f"{key} niqe", scores)
