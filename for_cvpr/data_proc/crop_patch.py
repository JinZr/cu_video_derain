import os
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm


def crop_image(
    img: Image.Image,
    save_path: str,
    img_name: str,
    patch_size: int = 128,
    stride: int = 64,
):
    img_arr = np.array(img)
    h, w, c = np.shape(img_arr)
    assert (h, w, c) == (1080, 1920, 3)
    assert len(img_name.split('.')) == 1, \
        f"img_name should not contain postfix like .{img_name.split('.')[-1]}"
    for x in range(w):
        for y in range(h):
            if x * stride + patch_size <= w and y * stride + patch_size <= h:
                patch_name = f"{img_name}_{x}_{y}.png"
                patch = img_arr[y * stride: y * stride + patch_size,
                                x * stride: x * stride + patch_size]
                Image.fromarray(patch).save(
                    fp=os.path.join(save_path, patch_name)
                )


def traverse_input_dir(root_dir: str, save_path: str):
    subdir_list = os.listdir(root_dir)
    for subdir in tqdm(subdir_list):
        subdir_path = os.path.join(root_dir, subdir)
        img_list = os.listdir(subdir_path)
        for img in img_list:
            img_name = img.split('.')[0]
            img_path = os.path.join(subdir_path, img)
            img_file = Image.open(img_path)
            img_save_path = os.path.join(save_path, subdir)
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)
            crop_image(
                img=img_file,
                save_path=img_save_path,
                img_name=img_name,
            )


def traverse_label_dir(root_dir: str, save_path: str):
    img_list = os.listdir(root_dir)
    for img in tqdm(img_list):
        img_name = img.split('.')[0]
        img_path = os.path.join(root_dir, img)
        img_file = Image.open(img_path)
        img_save_path = os.path.join(save_path, img_name)
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)
        crop_image(
            img=img_file,
            save_path=img_save_path,
            img_name=img_name,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--label_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--stride', type=int, default=64)

    args = parser.parse_args()

    input_path = args.input_path
    label_path = args.label_path
    save_path = args.save_path
    patch_size = args.patch_size
    stride = args.stride

    input_save_path = os.path.join(
        save_path, 'input', f'patch_size_{patch_size}_stride_{stride}')
    label_save_path = os.path.join(
        save_path, 'label', f'patch_size_{patch_size}_stride_{stride}')
    if not os.path.exists(input_save_path):
        os.makedirs(input_save_path)
    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path)

    print("== LABEL ==")
    traverse_label_dir(
        root_dir=label_path,
        save_path=label_save_path
    )

    print("== INPUT ==")
    traverse_input_dir(
        root_dir=input_path,
        save_path=input_save_path
    )
