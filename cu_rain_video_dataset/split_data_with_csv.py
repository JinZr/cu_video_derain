import os
import csv
import shutil

ROOT_PATH = '/Volumes/GoogleDrive-111940447000334462549/My Drive/cu_rain_video_dataset'
TRAIN_CSV = './train.csv'
TEST_CSV = './test.csv'


def parse_csv(path: str):
    res = []
    with open(path, 'r') as cin:
        for row in csv.reader(cin):
            res.append(row[1])
    return res[1:]


if __name__ == '__main__':

    train_set = parse_csv(TRAIN_CSV)
    test_set = parse_csv(TEST_CSV)
    print(train_set[:10], len(train_set))
    print(test_set[:10], len(test_set))

    root_path_dir = os.listdir(ROOT_PATH)
    print(root_path_dir)
    assert len(root_path_dir) == 3

    train_dir = os.path.join(ROOT_PATH, 'train')
    test_dir = os.path.join(ROOT_PATH, 'test')
    os.makedirs(train_dir)
    os.makedirs(test_dir)

    for dir in root_path_dir:
        print(f"== {dir} ==")
        current_src_path = os.path.join(ROOT_PATH, dir)
        current_train_dir = os.path.join(train_dir, dir)
        current_test_dir = os.path.join(test_dir, dir)

        if not os.path.exists(current_train_dir):
            os.makedirs(current_train_dir)
        if not os.path.exists(current_test_dir):
            os.makedirs(current_test_dir)

        obj_list = os.listdir(current_src_path)
        for obj in obj_list:
            obj_path = os.path.join(current_src_path, obj)
            if ".DS" in obj_path:
                continue
            if obj.split('.')[0].split('_')[0] in train_set:
                shutil.copy(obj_path, current_train_dir)
            elif obj.split('.')[0].split('_')[0] in test_set:
                shutil.copy(obj_path, current_test_dir)
            else:
                print(obj_path)
                raise NotImplementedError()
