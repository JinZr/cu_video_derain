import os
import json
import argparse

import numpy as np
from tqdm import tqdm
from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, required=True)
# parser.add_argument('--out_path', type=str, required=True)
args = parser.parse_args()

in_path = args.in_path
# out_path = args.out_path

annotation_path = os.path.join(in_path, 'Annotations/{}/{}.png')
pic_path = os.path.join(in_path, 'JPEGImages/{}/{}.jpg')

if __name__ == '__main__':
    assert os.path.exists(in_path)
    # assert not os.path.exists(out_path)
    # os.makedirs(out_path)
    js_file = os.path.join(in_path, 'meta.json')
    metadata_js = json.load(open(js_file))
    metadata = metadata_js['videos']
    for key in tqdm(metadata.keys()):
        vid_dict = metadata[key]
        obj_dict = vid_dict['objects']
        for obj_key in obj_dict.keys():
            obj = obj_dict[obj_key]
            frames = obj['frames']
            category = obj['category']
            for frame in frames:
                annotation = annotation_path.format(key, frame)
                pic = pic_path.format(key, frame)
                annotation = np.array(
                    np.array(Image.open(annotation), dtype=np.bool_),
                    dtype=np.int8
                )
                annotation_for_objv = np.stack(
                    [annotation, annotation, annotation], axis=-1)
                annotation_for_rain = 1 - annotation_for_objv
                pic = np.array(Image.open(pic))
                plt.imshow(annotation_for_objv * pic)
                plt.show()
