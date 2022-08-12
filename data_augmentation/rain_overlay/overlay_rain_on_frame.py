import os
import argparse
import random

import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt


def sort_list(video_frame_list):
    return sorted(
        video_frame_list,
        key=lambda x: int(x.split('.')[0].split('_')[-1])
    )


def green_blue_swap(image):
    import cv2
    # 3-channel image (no transparency)
    if image.shape[2] == 3:
        b, g, r = cv2.split(image)
        image[:, :, 0] = g
        image[:, :, 1] = b
    # 4-channel image (with transparency)
    elif image.shape[2] == 4:
        b, g, r, a = cv2.split(image)
        image[:, :, 0] = g
        image[:, :, 1] = b
    return image


parser = argparse.ArgumentParser()
parser.add_argument('--frame_dir', type=str, required=True)
# default='/home/desc/projects/derain/cu_rain_video_dataset/train/motion_regular_clip_frame')
parser.add_argument('--streak_dir', type=str, required=True)
# default='/home/desc/projects/derain/cu_rain_video_dataset/train/rain_streak_filtered')
parser.add_argument('--output_dir', type=str, required=True)
# default='/home/desc/projects/derain/cu_rain_video_dataset/train/motion_regular_augmented')
args = parser.parse_args()

frame_dir = args.frame_dir
streak_dir = args.streak_dir
output_dir = args.output_dir


def filter_out_garbage(in_list):
    return list(filter(lambda x: '._' not in x and '.DS_' not in x, in_list))


if __name__ == '__main__':
    frame_dir_list = sorted(os.listdir(frame_dir))
    streak_dir_list = os.listdir(streak_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for vid_dir in tqdm(frame_dir_list):

        K = random.randint(a=3, b=5)
        alphas = [random.uniform(a=0.8, b=1.2) for _ in range(K)]

        vid_path = os.path.join(frame_dir, vid_dir)
        output_path = os.path.join(output_dir, vid_dir)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        frame_list = os.listdir(vid_path)
        frame_list = filter_out_garbage(frame_list)
        frame_list = sort_list(frame_list)

        frame_paths = list(
            map(lambda x: os.path.join(vid_path, x), frame_list))
        frame_count = len(frame_list)
        selected_streak_dirs = random.choices(
            filter_out_garbage(os.listdir(streak_dir)), k=K)

        selected_streak_dir_paths = []
        selected_streak_frame_paths = []

        for selected_streak_dir in selected_streak_dirs:
            selected_streak_dir_paths.append(
                os.path.join(streak_dir, selected_streak_dir)
            )

        for selected_streak_dir_path in selected_streak_dir_paths:
            selected_streak_dir_path = os.path.join(
                streak_dir, selected_streak_dir_path
            )
            streak_frame_list = filter_out_garbage(os.listdir(
                selected_streak_dir_path)[:frame_count])
            selected_streak_frame_paths.append(
                sort_list(list(map(lambda x: os.path.join(
                    selected_streak_dir_path, x), streak_frame_list)))
            )

        rain_streak_index = 0
        ended_rain_streak = 0
        for frame_index in tqdm(range(frame_count)):
            frame = np.array(Image.open(frame_paths[frame_index]))
            streak_frames = []
            for rain_streak_arr_index, streak_frame in enumerate(selected_streak_frame_paths):
                try:
                    streak_frames.append(
                        np.array(Image.open(
                            streak_frame[rain_streak_index])) * alphas[rain_streak_arr_index]
                    )
                except:
                    ended_rain_streak += 1
                    pass
            if K - ended_rain_streak <= 1:
                break
            rain_streak_index += 1
            compressed_frame = np.sum(
                [frame] + streak_frames,
                axis=0,
            )
            compressed_frame = np.clip(
                a=compressed_frame,
                a_min=0,
                a_max=255,
            ).astype(np.uint8)
            compressed_frame = Image.fromarray(compressed_frame)
            compressed_frame.save(os.path.join(
                output_path, frame_list[frame_index]))
            # plt.imshow(compressed_frame)
            # plt.show()
        # exit(0)
