import os
import argparse

import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--video_pth', type=str, required=True)
parser.add_argument('--frame_dir', type=str, required=True)
args = parser.parse_args()
video_pth = args.video_pth
frame_dir = args.frame_dir


def vid_to_img(vid_path: str, img_seq_path: str):
    vidcap = cv2.VideoCapture(vid_path)
    filename = os.path.split(vid_path)[-1].split('.')[0]
    success, image = vidcap.read()
    count = 0
    success = True
    if not os.path.exists(os.path.join(img_seq_path, filename)):
        os.makedirs(os.path.join(img_seq_path, filename))
    while success:
        success, image = vidcap.read()
        if not success:
            break
        cv2.imwrite(
            os.path.join(img_seq_path, filename,
                         "frame_{}.png".format(count)), image
        )
        count += 1


if __name__ == '__main__':
    vid_to_img(
        vid_path=video_pth,
        img_seq_path=frame_dir
    )
