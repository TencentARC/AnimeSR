# :computer: How to Train AnimeSR

- [Overview](#overview)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
  - [Training step 1](#training-step-1)
  - [Training step 2](#training-step-2)
  - [Training step 3](#training-step-3)
- [The Pre-Trained Checkpoints](#the-pre-trained-checkpoints)
- [Other Tips](#other-tips)
    - [How to build your own (training) dataset ？](#how-to-build-your-own-training-dataset-)


## Overview
The training has been divided into three steps.
1. Training a Video Super-Resolution (VSR) model with a degradation model that only contains the classic basic operators (*i.e.*, blur, noise, downscale, compression).
2. Training several **L**earnable **B**asic **O**perators (**LBO**s). Using the VSR model from step 1 and the input-rescaling strategy to generate pseudo HR for real-world LR. The paired pseudo HR-LR data are used to train the LBO in a supervised manner.
3. Training the final VSR model with a degradation model containing both classic basic operators and learnable basic operators.

Specifically, the model training in each step consists of two stages. In the first stage, the model is trained with L1 loss from scratch. In the second stage, the model is fine-tuned with the combination of L1 loss, perceptual loss, and GAN loss.

## Dataset Preparation
We use AVC-Train dataset for our training. The AVC dataset is released under request, please refer to [Request for AVC-Dataset](README.md#request-for-avc-dataset).
After you download the AVC-Train dataset, put all the clips into one folder (dataset root). The dataset root should contain 553 folders(clips), each folder contains 100 frames, from `00000000.png` to `00000099.png`.

If you want to build your own (training) dataset or enlarge AVC-Train dataset, please refer to [How to build your own (training) dataset](#how-to-build-your-own-training-dataset).

## Training
As described in the paper, all the training is performed on four NVIDIA A100 GPUs in an internal cluster. You may need to adjust the batchsize according to the CUDA memory of your GPU card.

Before the training, you should modify the [option files](options/) accordingly. For example, you should modify the `dataroot_gt` to your own dataset root. We have comment all the lines you should modify with the `TO_MODIFY` tag.
### Training step 1
1. Train `Net` model

   Before the training, you should modify the [yaml file](options/train_animesr_step1_net_BasicOPonly.yml) accordingly. For example, you should modify the `dataroot_gt` to your own dataset root.
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 realanimevsr/train.py -opt options/train_animesr_step1_net_BasicOPonly.yml --launcher pytorch --auto_resume
   ```
2. Train `GAN` model

   The GAN model is fine-tuned from the `Net` model, as specified in `pretrain_network_g`
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 realanimevsr/train.py -opt options/train_animesr_step1_gan_BasicOPonly.yml --launcher pytorch --auto_resume
   ```

### Training step 2
The input frames for training LBO in the paper are included in the AVC dataset download link we sent you. These frames came from three real-world LR animation videos, and ~2,000 frames are selected from each video.

In order to obtain the paired data required for training LBO, you will need to use the VSR model obtained in step 1 and the input-rescaling strategy as described in the paper to process these input frames to obtain pseudo HR-LR paired data.
```bash
# make the soft link for the VSR model obtained in step 1
ln -s experiments/train_animesr_step1_net_BasicOPonly/models/net_g_300000.pth weights/step1_vsr_gan_model.pth
# using input-rescaling strategy to inference
python scripts/inference_animesr_frames.py -i datasets/lbo_training_data/real_world_video_to_train_lbo_1 -n step1_vsr_gan_model --input_rescaling_factor 0.5 --mod_scale 8 --expname input_rescaling_strategy_lbo_1
```
After the inference, you can train the LBO. Note that we only provide one [option file](options/train_animesr_step2_lbo_1_net.yml) for training `Net` model and one [option file](options/train_animesr_step2_lbo_1_gan.yml) for training `GAN` model. If you want to train multiple LBOs, just copy&paste those option files and modify the `name`, `dataroot_gt`, and `dataroot_lq`.
```bash
# train Net model
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 realanimevsr/train.py -opt options/train_animesr_step2_lbo_1_net.yml --launcher pytorch --auto_resume
# train GAN model
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 realanimevsr/train.py -opt options/train_animesr_step2_lbo_1_gan.yml --launcher pytorch --auto_resume
```


### Training step 3
Before the training, you will need to modify the `degradation_model_path` to the pre-trained LBO path.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 realanimevsr/train.py -opt options/train_animesr_step3_gan_3LBOs.yml --launcher pytorch --auto_resume
```

## Evaluation
See [evaluation readme](scripts/metrics/README.md).

## The Pre-Trained Checkpoints
You can download the checkpoints of all steps in [google drive](https://drive.google.com/drive/folders/1hCXhKNZYBADXsS_weHO2z3HhNE-Eg_jw?usp=share_link).


## Other Tips
#### How to build your own (training) dataset ？
Suppose you have a batch of HQ (high resolution, high bitrate, high quality) animation video, we provide the [anime_videos_preprocessing.py](scripts/anime_videos_preprocessing.py) script to help you to prepare training clips from the raw videos.

The preprocessing consists of 6 steps:
1. use FFmpeg to extract frames. Note that this step will take up a lot of disk space.
2. shot detection using [PySceneDetect](https://github.com/Breakthrough/PySceneDetect/)
3. flow estimation using spynet
4. black frames detection
5. image quality assessment using hyperIQA
6. generate clips for each video

```console
Usage: python scripts/anime_videos_preprocessing.py --dataroot datasets/YOUR_OWN_ANIME --n_thread 4 --run 1
  --dataroot           dataset root, dataroot/raw_videos should contains your HQ videos to be processed
  --n_thread           number of workers to process in parallel
  --run                which step to run. Since each step may take a long time, we recommend performing it step by step.
                       And after each step, check whether the output files are as expected
  --n_frames_per_clip  number of frames per clip. Default 100. You can increase the number if you want more training data
  --n_clips_per_video  number of clips per video. Default 1.  You can increase the number if you want more training data
```
After you finish all the steps, you will get the clips in `dataroot/select_clips`
