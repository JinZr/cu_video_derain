def sort_list(video_frame_list):
    return sorted(
        video_frame_list,
        key=lambda x: int(x.split('.')[0].split('_')[-1])
    )
