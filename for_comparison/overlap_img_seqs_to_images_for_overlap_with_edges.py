import os
import argparse

from tqdm import tqdm
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir


def img_seq_to_vid(input_dir: str, output_dir: str):
    # output_vid = os.path.join(
    #     output_dir,
    #     '{}.mp4'.format(input_dir.split('/')[-1])
    # )
    foldername = input_dir.split('/')[-1]
    os.makedirs(os.path.join(output_dir, foldername))

    files = os.listdir(input_dir)
    files = list(filter(
        lambda x: os.path.isfile(os.path.join(input_dir, x)),
        files
    ))
    files = sorted(
        files,
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    img_arr = []
    with tqdm(total=len(files)) as pbar:
        for file in files:
            if file == ".DS_Store":
                continue
            pbar.update()
            # cv2.imread(os.path.join(input_dir, file))
            img_inf = Image.open(os.path.join(input_dir, file)).convert("RGB")
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
            img_arr.append(
                mean_frame
            )
    with tqdm(total=len(files)) as pbar:
        for idx, img in enumerate(img_arr):
            pbar.update()
            Image.fromarray(img).save(os.path.join(output_dir, foldername, '{}.png'.format(idx)))


if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dirs = os.listdir(input_dir)
    for dir in dirs:
        if dir == ".DS_Store":
                continue
        current_dir = os.path.join(input_dir, dir)
        img_seq_to_vid(
            input_dir=current_dir,
            output_dir=output_dir
        )
