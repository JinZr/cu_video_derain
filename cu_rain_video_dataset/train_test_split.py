import os
import csv
import random
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()

csv_file = args.csv_file
output_dir = args.output_dir

daytime_list = []
nighttime_list = []

HERDER = ['Num.', 'Fname.', 'Time', 'Density', 'Scene']

if __name__ == '__main__':
    assert os.path.exists(output_dir) and os.path.isdir(output_dir)
    assert os.path.exists(csv_file) and os.path.isfile(csv_file)
    with open(csv_file, 'r') as csv_in:
        reader = csv.reader(csv_in)
        for row in tqdm(reader):
            if row[2] == 'daytime':
                daytime_list.append(row[:-1])
            elif row[2] == 'nighttime':
                nighttime_list.append(row[:-1])
        assert len(daytime_list) + len(nighttime_list) == reader.line_num - 1

    print(f"num. of daytime video: {len(daytime_list)}")
    print(f"num. of nighttime video: {len(nighttime_list)}")
    print(f"num. of video: \t{len(nighttime_list) + len(daytime_list)}")
    selected_nighttime_list = random.sample(
        population=nighttime_list, k=15)
    selected_daytime_list = random.sample(
        population=daytime_list, k=35)
    selected_train_list = []
    with open(csv_file, 'r') as csv_in:
        reader = csv.reader(csv_in)
        for row in tqdm(reader):
            r = row[:-1]
            if r not in selected_daytime_list and r not in selected_nighttime_list:
                selected_train_list.append(r)
    assert len(selected_train_list) + len(selected_daytime_list) + \
        len(selected_nighttime_list) == reader.line_num
    with open(f"{output_dir}/test_daytime.csv", 'w+') as daytime_csv:
        writer = csv.writer(daytime_csv)
        writer.writerow(HERDER)
        writer.writerows(selected_daytime_list)
    with open(f"{output_dir}/test_nighttime.csv", 'w+') as nighttime_csv:
        writer = csv.writer(nighttime_csv)
        writer.writerow(HERDER)
        writer.writerows(selected_nighttime_list)
    with open(f"{output_dir}/train.csv", 'w+') as train_csv:
        writer = csv.writer(train_csv)
        writer.writerows(selected_train_list)
    with open(f"{output_dir}/test.csv", 'w+') as test_csv:
        writer = csv.writer(test_csv)
        writer.writerow(HERDER)
        writer.writerows(selected_daytime_list + selected_nighttime_list)
