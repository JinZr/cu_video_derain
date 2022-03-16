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
    category_dir_list = os.listdir(frame_dir)
    res = []
    for cat_dir in category_dir_list:
        cat_path = os.path.join(frame_dir, cat_dir)
        vid_list = os.listdir(cat_path)
        for vid_dir in vid_list:
            vid_path = os.path.join(cat_path, vid_dir)
            frame_list = os.listdir(vid_path)
            frame_list = sorted(
                frame_list,
                key=lambda x: int(x.split('.')[0].split('_')[-1]))
            tmp = []
            for frame in frame_list:
                frame_path = os.path.join(vid_path, frame)
                tmp.append({
                    'gt': frame_path,
                    'rain': frame_path
                })
            res.append(tmp)

    json_str = json.dumps(res, indent=4)
    with open(output_json, 'w+') as fout:
        fout.write(json_str)
