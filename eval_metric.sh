# Input
python eval_psnr_ssim.py --gt_path=/home/desc/Downloads/video/test_heavy/final-static-img/ --inf_path=/home/desc/Downloads/video/test_heavy_frames

python eval_psnr_ssim.py --gt_path=/home/desc/Downloads/video/test_light/final-static-img/ --inf_path=/home/desc/Downloads/video/test_light_frames

# IDT
python eval_psnr_ssim.py --gt_path=/home/desc/Downloads/video/test_heavy/final-static-img/ --inf_path=/home/desc/projects/derain/2021/IDT/result_ours/RainSPA/heavy

python eval_psnr_ssim.py --gt_path=/home/desc/Downloads/video/test_light/final-static-img/ --inf_path=/home/desc/projects/derain/2021/IDT/result_ours/RainSPA/light

# ESTINet
python eval_psnr_ssim_resize.py --gt_path=/home/desc/Downloads/video/test_heavy/final-static-img/ --inf_path=/home/desc/projects/derain/2021/Enhanced-Spatio-Temporal-Interaction-Learning-for-Video-Deraining/out/heavy/F_out/

python eval_psnr_ssim_resize.py --gt_path=/home/desc/Downloads/video/test_light/final-static-img/ --inf_path=/home/desc/projects/derain/2021/Enhanced-Spatio-Temporal-Interaction-Learning-for-Video-Deraining/out/light/F_out/

# ESTINet Retrained
python eval_psnr_ssim_for_overlap.py --gt_path=/home/desc/Downloads/video/test_heavy/final-static-img/ --inf_path=/home/desc/projects/derain/2021/Enhanced-Spatio-Temporal-Interaction-Learning-for-Video-Deraining/out/cuhk_heavy_retrain

python eval_psnr_ssim_for_overlap.py --gt_path=/home/desc/Downloads/video/test_light/final-static-img/ --inf_path=/home/desc/projects/derain/2021/Enhanced-Spatio-Temporal-Interaction-Learning-for-Video-Deraining/out/cuhk_light_retrain

# nafnet
python eval_psnr_ssim.py --gt_path=/home/desc/Downloads/video/test_heavy/final-static-img/ --inf_path=/data/projects/derain/2022/nafnet/CUHK_test_heavy/

python eval_psnr_ssim.py --gt_path=/home/desc/Downloads/video/test_light/final-static-img/ --inf_path=/data/projects/derain/2022/nafnet/CUHK_test_light/

# S2VD
python eval_psnr_ssim.py --gt_path=/home/desc/Downloads/video/test_heavy/final-static-img/ --inf_path=/data/projects/derain/2021/S2VD/S2VD-ours_final_heavy/

python eval_psnr_ssim.py --gt_path=/home/desc/Downloads/video/test_light/final-static-img/ --inf_path=/data/projects/derain/2021/S2VD/S2VD-ours_final_light/

# RDD-Net
python eval_psnr_ssim.py --gt_path=/home/desc/Downloads/video/test_heavy/final-static-img/ --inf_path=/data/projects/derain/2022/RDD-Net/Results/heavy_ours

python eval_psnr_ssim.py --gt_path=/home/desc/Downloads/video/test_light/final-static-img/ --inf_path=/data/projects/derain/2022/RDD-Net/Results/light_ours

# Ours
python eval_psnr_ssim_for_overlap_with_edge.py --gt_path=/home/desc/Downloads/video/test_heavy/final-static-img/ --inf_path=/data/projects/derain/Ours/cuhk-heavy

python eval_psnr_ssim_for_overlap_with_edge.py --gt_path=/home/desc/Downloads/video/test_light/final-static-img/ --inf_path=/data/projects/derain/Ours/cuhk-light
