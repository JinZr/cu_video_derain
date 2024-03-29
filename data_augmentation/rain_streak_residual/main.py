import os
import argparse

import cv2
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--frames_pth', type=str, required=True)
parser.add_argument('--static_pth', type=str, required=True)
parser.add_argument('--overlay_pth', type=str, required=True)
args = parser.parse_args()

frames_pth = args.frames_pth
static_pth = args.static_pth
overlay_pth = args.overlay_pth

if __name__ == '__main__':
    frame_dir_list = os.listdir(frames_pth)
    for frame_dir in frame_dir_list:
        print(f"==== {frame_dir} ====")
        static_image_path = os.path.join(static_pth, f"{frame_dir}.png")
        static_image = cv2.imread(static_image_path)

        video_frames_path = os.path.join(frames_pth, frame_dir)
        video_frame_list = os.listdir(video_frames_path)
        video_frame_list = sorted(
            video_frame_list,
            key=lambda x: int(x.split('.')[0].split('_')[-1])
        )

        frame_overlay_pth = os.path.join(overlay_pth, frame_dir)
        if not os.path.exists(frame_overlay_pth):
            os.makedirs(frame_overlay_pth)

        for frame in tqdm(video_frame_list):
            output_streak_path = os.path.join(frame_overlay_pth, frame)
            frame_path = os.path.join(video_frames_path, frame)
            current_frame = cv2.imread(frame_path)

            rain_streak = current_frame.astype(
                np.float32) - static_image.astype(np.float32)
            # rain_streak = cv2.threshold(rain_streak, thresh=200, maxval=254, type=3)[1]
            # rain_streak = cv2.inRange(
            #     src=rain_streak,
            #     lowerb=(200, 200, 200),
            #     upperb=(255, 255, 255)
            # )

            # print(rain_streak)
            cv2.imwrite(output_streak_path, rain_streak)
