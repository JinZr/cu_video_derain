import os
import argparse

import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_vid', type=str, required=True)
args = parser.parse_args()

input_dir = args.input_dir
output_vid = args.output_vid

if __name__ == '__main__':
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
