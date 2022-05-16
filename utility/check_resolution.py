import os
import argparse

from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()

path = args.path

H, W = 1080, 1920

ERROR_LIST = []


def check_resolution(path: str):
    ls = os.listdir(path=path)
    for l in tqdm(ls):
        l_path = os.path.join(path, l)
        if os.path.isdir(l_path):
            check_resolution(l_path)
        elif '.png' in l_path:
            img = Image.open(l_path)
            if img.width == W and img.height == H:
                pass
            else:
                ERROR_LIST.append(l_path + '\n')


if __name__ == '__main__':
    check_resolution(path=path)
    with open('./error.list', 'w+') as fout:
        fout.writelines(ERROR_LIST)
