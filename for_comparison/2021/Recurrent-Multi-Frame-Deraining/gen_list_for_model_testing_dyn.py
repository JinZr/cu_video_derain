# generate a list of folders, each folder is associated with a video
# and contains frames extracted from the video

import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--frame_dir', type=str, required=True)
parser.add_argument('--output_list', type=str, required=True)
args = parser.parse_args()

frame_dir = args.frame_dir
output_list = args.output_list

if __name__ == '__main__':
    catlist = os.listdir(frame_dir)
    res = []
    for cat_dir in catlist:
        cat_path = os.path.join(frame_dir, cat_dir)
        vid_list = os.listdir(cat_path)
        for vid_dir in vid_list:
            vid_path = os.path.join(cat_path, vid_dir)
            res.append(f"{cat_dir}/{vid_dir}")
    with open(output_list, 'w+') as fout:
        fout.writelines(list(map(
            lambda x: x + '\n',
            res
        )))
