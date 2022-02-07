import os
import argparse

import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--input_folder', type=str, required=True)
# /home/desc/projects/derain/2021/S2VD/my_dataset
parser.add_argument('--output_folder', type=str, required=True)
# /home/desc/projects/derain/2021/S2VD/my_dataset_fixed

args = parser.parse_args()
input_folder = args.input_folder
output_folder = args.output_folder


def count_files(path: str) -> int:
    ct = 0
    dirs = os.listdir(path)
    for dir in dirs:
        ct += len(os.listdir(os.path.join(path, dir)))
    return ct


if __name__ == '__main__':
    with tqdm(total=count_files(input_folder)) as pbar:
        dirs = os.listdir(input_folder)
        for dir in dirs:
            current_input_dir = os.path.join(input_folder, dir)
            current_output_dir = os.path.join(output_folder, dir)
            if not os.path.exists(current_output_dir):
                os.makedirs(current_output_dir)
            files = os.listdir(current_input_dir)
            for file in files:
                pbar.update()
                current_input_file_path = os.path.join(current_input_dir, file)
                current_output_file_path = os.path.join(
                    current_output_dir, file)
                input_img = cv2.imread(current_input_file_path)
                R, G, B = cv2.split(input_img)
                output_img = cv2.merge([R, G, B])
                cv2.imwrite(
                    current_output_file_path,
                    output_img,
                )
