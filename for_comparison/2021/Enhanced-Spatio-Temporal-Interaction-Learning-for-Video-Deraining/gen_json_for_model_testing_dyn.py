"""
    JSON file format: [[
        {
            "rf": path_to_rain_frame,
            "gt": path_to_rain_free_frame
        },
        {
            "rf": path_to_rain_frame,
            "gt": path_to_rain_free_frame
        }
    ]]
"""
import os
import json
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--frame_dir', type=str, required=True)
parser.add_argument('--output_json', type=str, required=True)

args = parser.parse_args()
frame_dir = args.frame_dir
output_json = args.output_json

if __name__ == '__main__':
    video_dir_list = os.listdir(frame_dir)
    res = []
    for video_dir in video_dir_list:
        print(f"{video_dir}")
        video_path = os.path.join(frame_dir, video_dir)
        frame_list = os.listdir(video_path)
        frame_list = sorted(
            frame_list,
            key=lambda x: int(x.split('.')[0].split('_')[-1]))
        frame_list = list(
            map(lambda x: os.path.join(video_path, x), frame_list))
        tmp = []
        for frame in tqdm(frame_list):
            frame = str(frame)
            tmp.append({
                'gt': frame.replace("motion_100_augmented", "motion_100"),
                'rain': frame
            })
        res.append(tmp)
    json_str = json.dumps(res, indent=4)
    with open(output_json, 'w+') as fout:
        fout.write(json_str)
