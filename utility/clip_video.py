import os
import argparse

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--vid_dir', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)
args = parser.parse_args()

vid_dir = args.vid_dir
out_dir = args.out_dir


def clip_vid(input_dir: str, output_dir: str, vid_name: str):
    ffmpeg_extract_subclip(
        f"{input_dir}/{vid_name}", 0, 5,
        targetname=f"{output_dir}/{vid_name}"
    )


if __name__ == '__main__':
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    vid_list = os.listdir(vid_dir)
    for vid_name in tqdm(vid_list):
        try:
            clip_vid(
                input_dir=vid_dir,
                output_dir=out_dir,
                vid_name=vid_name
            )
        except:
            print(vid_name)
