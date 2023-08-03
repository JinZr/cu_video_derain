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

parser.add_argument("--frame_dir", type=str, required=True)
parser.add_argument("--gt_dir", type=str, required=True)
parser.add_argument("--output_json", type=str, required=True)

args = parser.parse_args()
frame_dir = args.frame_dir
gt_dir = args.gt_dir
output_json = args.output_json


def filter_out_garbage(in_list):
    return list(filter(lambda x: "._" not in x and ".DS_" not in x, in_list))


if __name__ == "__main__":
    gt_file_list = os.listdir(gt_dir)
    res = []
    for gt_filename in tqdm(gt_file_list):
        gt_filepath = os.path.join(gt_dir, gt_filename)
        gt_pure_filename = gt_filename.split(".")[0]

        fm_dir = os.path.join(frame_dir, gt_pure_filename)
        print(fm_dir)
        fm_file_list = sorted(
            filter_out_garbage(os.listdir(fm_dir)),
            key=lambda x: int(x.split(".")[0].split("_")[-1]),
        )

        fm_list = []
        for fm_filename in tqdm(fm_file_list):
            fm_filepath = os.path.join(fm_dir, fm_filename)
            fm_list.append({"rain": fm_filepath, "gt": fm_filepath})
        res.append(fm_list)
    json_str = json.dumps(res, indent=4)
    with open(output_json, "w+") as fout:
        fout.write(json_str)
