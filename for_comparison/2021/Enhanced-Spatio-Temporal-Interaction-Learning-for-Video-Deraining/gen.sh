python gen_json_for_model_testing.py \
    --frame_dir=/home/desc/projects/derain/cu_rain_video_dataset/train/final-static-vid_frame/ \
    --gt_dir=/home/desc/projects/derain/cu_rain_video_dataset/train/final-static-img \
    --output_json=/home/desc/projects/derain/2021/Enhanced-Spatio-Temporal-Interaction-Learning-for-Video-Deraining/code/data/cu_static_train.json

# python gen_json_for_model_testing.py \
#     --frame_dir=/home/desc/projects/derain/cu_rain_video_dataset/test/final-static-vid_frame/ \
#     --gt_dir=/home/desc/projects/derain/cu_rain_video_dataset/test/final-static-img \
#     --output_json=/home/desc/projects/derain/2021/Enhanced-Spatio-Temporal-Interaction-Learning-for-Video-Deraining/code/data/cu_static_test.json

# python gen_json_for_motion_videos.py \
#     --frame_dir=/home/desc/projects/derain/cu_rain_video_dataset/train/motion_regular_augmented/ \
#     --gt_dir=/home/desc/projects/derain/cu_rain_video_dataset/train/motion_regular_clip_frame/ \
#     --output_json=/home/desc/projects/derain/2021/Enhanced-Spatio-Temporal-Interaction-Learning-for-Video-Deraining/code/data/cu_dynamic_train.json

python gen_json_for_motion_videos.py \
    --frame_dir=/home/desc/projects/derain/cu_rain_video_dataset/train/shake_processed_frame_augmented/ \
    --gt_dir=/home/desc/projects/derain/cu_rain_video_dataset/train/shake_processed_frame/ \
    --output_json=/home/desc/projects/derain/2021/Enhanced-Spatio-Temporal-Interaction-Learning-for-Video-Deraining/code/data/cu_dynamic_extra_train.json
