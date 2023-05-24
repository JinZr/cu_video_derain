import os
import argparse

import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir


def img_seq_to_vid(input_dir: str, output_dir: str):
    output_vid = os.path.join(
        output_dir,
        '{}.mp4'.format(input_dir.split('/')[-1])
    )
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
            pbar.update()
            img_arr.append(
                cv2.imread(os.path.join(input_dir, file))
            )
    print('writing video ...')
    height, width, _ = img_arr[0].shape
    writer = cv2.VideoWriter(
        output_vid,
        cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
        30,
        (width, height)
    )
    with tqdm(total=len(files)) as pbar:
        for img in img_arr:
            pbar.update()
            writer.write(img)
    writer.release()


if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dirs = os.listdir(input_dir)
    for dir in dirs:
        if dir == ".DS_Store": continue
        current_dir = os.path.join(input_dir, dir)
        img_seq_to_vid(
            input_dir=current_dir,
            output_dir=output_dir
        )
