# AnimeSR (NeurIPS 2022)

### :open_book: AnimeSR: Learning Real-World Super-Resolution Models for Animation Videos
> [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2206.07038)<br>
> [Yanze Wu](https://github.com/ToTheBeginning), [Xintao Wang](https://xinntao.github.io/), [Gen Li](https://scholar.google.com.hk/citations?user=jBxlX7oAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> [Tencent ARC Lab](https://arc.tencent.com/en/index); Platform Technologies, Tencent Online Video


### :triangular_flag_on_post: Updates
* **2022.11.28**: release codes&models.
* **2022.08.29**: release AVC-Train and AVC-Test.


## Video Demos

https://user-images.githubusercontent.com/11482921/204205018-d69e2e51-fbdc-4766-8293-a40ffce3ed25.mp4

https://user-images.githubusercontent.com/11482921/204205109-35866094-fa7f-413b-8b43-bb479b42dfb6.mp4



## :wrench: Dependencies and Installation
- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- Other required packages in `requirements.txt`

### Installation

1. Clone repo

    ```bash
    git clone https://github.com/TencentARC/AnimeSR.git
    cd AnimeSR
    ```
2. Install

    ```bash
    # Install dependent packages
    pip install -r requirements.txt

    # Install AnimeSR
    python setup.py develop
    ```

## :zap: Quick Inference
Download the pre-trained AnimeSR models [[Google Drive](https://drive.google.com/drive/folders/1gwNTbKLUjt5FlgT6PQQnBz5wFzmNUX8g?usp=share_link)], and put them into the [weights](weights/) folder. Currently, the available pre-trained models are:
- `AnimeSR_v1-PaperModel.pth`: v1 model, also the paper model. You can use this model for paper results reproducing.
- `AnimeSR_v2.pth`: v2 model. Compare with v1, this version has better naturalness, fewer artifacts, and better texture/background restoration. If you want better results, use this model.

AnimeSR supports both frames and videos as input for inference. We provide several sample test cases in [google drive](https://drive.google.com/drive/folders/1K8JzGNqY_pHahYBv51iUUI7_hZXmamt2?usp=share_link), you can download it and put them to [inputs](inputs/) folder.

**Inference on Frames**
```bash
python scripts/inference_animesr_frames.py -i inputs/tom_and_jerry -n AnimeSR_v2 --expname animesr_v2 --save_video_too --fps 20
```
```console
Usage:
  -i --input           Input frames folder/root. Support first level dir (i.e., input/*.png) and second level dir (i.e., input/*/*.png)
  -n --model_name      AnimeSR model name. Default: AnimeSR_v2, can also be AnimeSR_v1-PaperModel
  -s --outscale        The netscale is x4, but you can achieve arbitrary output scale (e.g., x2 or x1) with the argument outscale.
                       The program will further perform cheap resize operation after the AnimeSR output. Default: 4
  -o --output          Output root. Default: results
  -expname             Identify the name of your current inference. The outputs will be saved in $output/$expname
  -save_video_too      Save the output frames to video. Default: off
  -fps                 The fps of the (possible) saved videos. Default: 24
```
After run the above command, you will get the SR frames in `results/animesr_v2/frames` and the SR video in `results/animesr_v2/videos`.

**Inference on Video**
```bash
# single gpu and single process inference
CUDA_VISIBLE_DEVICES=0 python scripts/inference_animesr_video.py -i inputs/TheMonkeyKing1965.mp4 -n AnimeSR_v2 -s 4 --expname animesr_v2 --num_process_per_gpu 1 --suffix 1gpu1process
# single gpu and multi process inference (you can use multi-processing to improve GPU utilization)
CUDA_VISIBLE_DEVICES=0 python scripts/inference_animesr_video.py -i inputs/TheMonkeyKing1965.mp4 -n AnimeSR_v2 -s 4 --expname animesr_v2 --num_process_per_gpu 3 --suffix 1gpu3process
# multi gpu and multi process inference
CUDA_VISIBLE_DEVICES=0,1 python scripts/inference_animesr_video.py -i inputs/TheMonkeyKing1965.mp4 -n AnimeSR_v2 -s 4 --expname animesr_v2 --num_process_per_gpu 3 --suffix 2gpu6process
```
```console
Usage:
  -i --input           Input video path or extracted frames folder
  -n --model_name      AnimeSR model name. Default: AnimeSR_v2, can also be AnimeSR_v1-PaperModel
  -s --outscale        The netscale is x4, but you can achieve arbitrary output scale (e.g., x2 or x1) with the argument outscale.
                       The program will further perform cheap resize operation after the AnimeSR output. Default: 4
  -o -output           Output root. Default: results
  -expname             Identify the name of your current inference. The outputs will be saved in $output/$expname
  -fps                 The fps of the (possible) saved videos. Default: None
  -extract_frame_first If input is a video, you can still extract the frames first, other wise AnimeSR will read from stream
  -num_process_per_gpu Since the slow I/O speed will make GPU utilization not high enough, so as long as the
                       video memory is sufficient, we recommend placing multiple processes on one GPU to increase the utilization of each GPU.
                       The total process will be number_process_per_gpu * num_gpu
  -suffix              You can add a suffix string to the sr video name, for example, 1gpu3processx2 which means the SR video is generated with one GPU and three process and the outscale is x2
  -half                Use half precision for inference, it won't make big impact on the visual results
```
SR videos are saved in `results/animesr_v2/videos/$video_name` folder.

If you are looking for portable executable files, you can try our [realesr-animevideov3](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_video_model.md) model which shares the similar technology with AnimeSR.



## :computer: Training
See [Training.md](Training.md)

## Request for AVC-Dataset
1. Download and carefully read the [LICENSE AGREEMENT](assets/LICENSE%20AGREEMENT.pdf) PDF file.
2. If you understand, acknowledge, and agree to all the terms specified in the [LICENSE AGREEMENT](assets/LICENSE%20AGREEMENT.pdf). Please email `yanzewu@tencent.com` with the **LICENSE AGREEMENT PDF** file, **your name**, and **institution**. We will keep the license and send the download link of AVC dataset to you.


## Acknowledgement
This project is build based on [BasicSR](https://github.com/XPixelGroup/BasicSR).

##  Citation
If you find this project useful for your research, please consider citing our paper:
```bibtex
@InProceedings{wu2022animesr,
  author={Wu, Yanze and Wang, Xintao and Li, Gen and Shan, Ying},
  title={AnimeSR: Learning Real-World Super-Resolution Models for Animation Videos},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## :e-mail: Contact
If you have any question, please email `yanzewu@tencent.com`.
