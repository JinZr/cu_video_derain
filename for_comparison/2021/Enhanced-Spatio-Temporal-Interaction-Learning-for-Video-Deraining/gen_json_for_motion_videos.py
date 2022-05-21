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
parser.add_argument('--gt_dir', type=str, required=True)
parser.add_argument('--output_json', type=str, required=True)

args = parser.parse_args()
frame_dir = args.frame_dir
gt_dir = args.gt_dir
output_json = args.output_json

if __name__ == '__main__':
    frame_dir_list = os.listdir(frame_dir)
    res = []
    for fdir in tqdm(frame_dir_list):
        frame_dir_path = os.path.join(frame_dir, fdir)
        frame_list = sorted(
            os.listdir(frame_dir_path),
            key=lambda x: int(x.split('.')[0].split('_')[-1])
        )
        fm_list = []
        for frame in frame_list:
            frame_path = os.path.join(frame_dir_path, frame)
            gt_frame_path = os.path.join(gt_dir, fdir, frame)
            fm_list.append({
                'rain': frame_path,
                'gt': gt_frame_path
            })
        res.append(fm_list)
    json_str = json.dumps(res, indent=4)
    with open(output_json, 'w+') as fout:
        fout.write(json_str)
