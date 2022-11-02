import argparse
import cv2
import glob
import numpy as np
import os
import shutil
import torch
import torchvision
from multiprocessing import Pool
from os import path as osp
from PIL import Image
from tqdm import tqdm

from animesr.utils import video_util
from animesr.utils.shot_detector import ShotDetector
from basicsr.archs.spynet_arch import SpyNet
from basicsr.utils import img2tensor
from basicsr.utils.download_util import download_file_from_google_drive
from facexlib.assessment import init_assessment_model


def main(args):
    """A script to prepare anime videos.

    The preparation can be divided into following steps:
    1. use ffmpeg to extract frames
    2. shot detection
    3. estimate flow
    4. detect black frames
    5. use hyperIQA to evaluate the quality of frames
    6. generate at most 5 clips per video
    """

    opt = dict()

    opt['debug'] = args.debug
    opt['n_thread'] = args.n_thread
    opt['ss_idx'] = args.ss_idx
    opt['to_idx'] = args.to_idx

    # params for step1: extract frames
    opt['video_root'] = f'{args.dataroot}/raw_videos'
    opt['save_frames_root'] = f'{args.dataroot}/frames'
    opt['meta_files_root'] = f'{args.dataroot}/meta'

    # params for step2: shot detection
    opt['detect_shot_root'] = f'{args.dataroot}/detect_shot'

    # params for step3: flow estimation
    opt['estimate_flow_root'] = f'{args.dataroot}/estimate_flow'
    opt['spy_pretrain_weight'] = 'experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth'
    opt['downscale_factor'] = 1

    # params for step4: detect black frames
    opt['black_flag_root'] = f'{args.dataroot}/black_flag'
    opt['black_threshold'] = 0.98

    # params for step5: image quality assessment
    opt['num_patch_per_iqa'] = 5
    opt['iqa_score_root'] = f'{args.dataroot}/iqa_score'

    # params for step6: generate clips
    opt['num_frames_per_clip'] = args.n_frames_per_clip
    opt['num_clips_per_video'] = args.n_clips_per_video
    opt['select_clips_root'] = f'{args.dataroot}/{args.select_clip_root}'
    opt['select_clips_meta'] = osp.join(opt['select_clips_root'], 'meta_info')
    opt['select_clips_frames'] = osp.join(opt['select_clips_root'], 'frames')
    opt['select_done_flags'] = osp.join(opt['select_clips_root'], 'done_flags')

    if '1' in args.run:
        run_step1(opt)
    if '2' in args.run:
        run_step2(opt)
    if '3' in args.run:
        run_step3(opt)
    if '4' in args.run:
        run_step4(opt)
    if '5' in args.run:
        run_step5(opt)
    if '6' in args.run:
        run_step6(opt)


# -------------------------------------------------------------------- #
# --------------------------- step1 ---------------------------------- #
# -------------------------------------------------------------------- #


def run_step1(opt):
    """extract frames

    1. read all video files under video_root folder
    2. filter out the videos that already have been processed
    3. use multi-process to extract the remaining videos

    """

    video_root = opt['video_root']
    frames_root = opt['save_frames_root']
    meta_root = opt['meta_files_root']
    os.makedirs(frames_root, exist_ok=True)
    os.makedirs(meta_root, exist_ok=True)

    if not osp.isdir(video_root):
        print(f'path {video_root} is not a valid folder, exit.')

    videos_path = sorted(glob.glob(osp.join(video_root, '*')))
    if opt['debug']:
        videos_path = videos_path[:3]
    else:
        videos_path = videos_path[opt['ss_idx']:opt['to_idx']]
    pbar = tqdm(total=len(videos_path), unit='video', desc='step1')
    pool = Pool(opt['n_thread'])
    for video_path in videos_path:
        video_name = osp.splitext(osp.basename(video_path))[0]
        if video_name.startswith('.'):
            print(f'skip {video_name}')
            continue
        frame_path = osp.join(frames_root, video_name)
        meta_path = osp.join(meta_root, f'{video_name}.txt')
        pool.apply_async(
            worker1, args=(opt, video_name, video_path, frame_path, meta_path), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()


def worker1(opt, video_name, video_path, frame_path, meta_path):
    # get info of video
    fps = video_util.get_video_fps(video_path)
    h, w = video_util.get_video_resolution(video_path)
    num_frames = video_util.get_video_num_frames(video_path)
    bit_rate = video_util.get_video_bitrate(video_path)

    # check whether this video has been processed
    flag = True
    num_extracted_frames = 0
    if osp.exists(frame_path):
        num_extracted_frames = len(glob.glob(osp.join(frame_path, '*.png')))
        if num_extracted_frames == num_frames:
            print(f'skip {video_path} since there are already {num_frames} frames have been extracted.')
            flag = False
        else:
            print(f'{num_extracted_frames} of {num_frames} have been extracted for {video_path}, re-run.')

    # extract frames
    os.makedirs(frame_path, exist_ok=True)
    video_util.video2frames(video_path, frame_path, force=flag, high_quality=True)
    if flag:
        num_extracted_frames = len(glob.glob(osp.join(frame_path, '*.png')))

    # write some metadata to meta file
    with open(meta_path, 'w') as f:
        f.write(f'Video Name: {video_name}\n')
        f.write(f'H: {h}\n')
        f.write(f'W: {w}\n')
        f.write(f'FPS: {fps}\n')
        f.write(f'Bit Rate: {bit_rate}kbps\n')
        f.write(f'{num_extracted_frames}/{num_frames} have been extracted\n')


# -------------------------------------------------------------------- #
# --------------------------- step2 ---------------------------------- #
# -------------------------------------------------------------------- #


def run_step2(opt):
    """shot detection. refer to lijian's pipeline"""
    detect_shot_root = opt['detect_shot_root']
    meta_root = opt['meta_files_root']
    os.makedirs(detect_shot_root, exist_ok=True)
    if not osp.exists(meta_root):
        print('no videos has run step1, exit.')
        return

    # get the video which has been extracted frames
    videos_name = sorted(glob.glob(osp.join(meta_root, '*.txt')))
    videos_name = [osp.splitext(osp.basename(video_name))[0] for video_name in videos_name]

    if opt['debug']:
        videos_name = videos_name[:3]
    else:
        videos_name = videos_name[opt['ss_idx']:opt['to_idx']]

    pbar = tqdm(total=len(videos_name), unit='video', desc='step2')
    pool = Pool(opt['n_thread'])
    for video_name in videos_name:
        pool.apply_async(worker2, args=(opt, video_name), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()


def worker2(opt, video_name):
    video_frame_path = osp.join(opt['save_frames_root'], video_name)
    detect_shot_file_path = osp.join(opt['detect_shot_root'], f'{video_name}.txt')
    if osp.exists(detect_shot_file_path):
        print(f'skip {video_name} since {detect_shot_file_path} already exist.')
        return

    detector = ShotDetector()
    shot_list = detector.detect_shots(video_frame_path)
    with open(detect_shot_file_path, 'w') as f:
        for shot in shot_list:
            f.write(f'{shot[0]} {shot[1]}\n')


# -------------------------------------------------------------------- #
# --------------------------- step3 ---------------------------------- #
# -------------------------------------------------------------------- #


def run_step3(opt):
    estimate_flow_root = opt['estimate_flow_root']
    meta_root = opt['meta_files_root']
    os.makedirs(estimate_flow_root, exist_ok=True)
    if not osp.exists(meta_root):
        print('no videos has run step1, exit.')
        return

    # download the spynet checkpoint first
    if not osp.exists(opt['spy_pretrain_weight']):
        download_file_from_google_drive('1VZz1cikwTRVX7zXoD247DB7n5Tj_LQpF', opt['spy_pretrain_weight'])

    # get the video which has been extracted frames
    videos_name = sorted(glob.glob(osp.join(meta_root, '*.txt')))
    videos_name = [osp.splitext(osp.basename(video_name))[0] for video_name in videos_name]

    if opt['debug']:
        videos_name = videos_name[:3]
    else:
        videos_name = videos_name[opt['ss_idx']:opt['to_idx']]

    pbar = tqdm(total=len(videos_name), unit='video', desc='step3')

    num_gpus = torch.cuda.device_count()
    ctx = torch.multiprocessing.get_context('spawn')
    pool = ctx.Pool(min(3 * num_gpus, opt['n_thread']))
    for idx, video_name in enumerate(videos_name):
        pool.apply_async(
            worker3, args=(opt, video_name, torch.device(idx % num_gpus)), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()


def read_img(img_path, device, downscale_factor=1):
    img = cv2.imread(img_path)
    h, w = img.shape[0:2]
    if downscale_factor != 1:
        img = cv2.resize(img, (w // downscale_factor, h // downscale_factor), interpolation=cv2.INTER_LANCZOS4)
    img = img2tensor(img)
    img = img.unsqueeze(0).to(device)
    return img


@torch.no_grad()
def worker3(opt, video_name, device):
    video_frame_path = osp.join(opt['save_frames_root'], video_name)
    frames_path = sorted(glob.glob(osp.join(video_frame_path, '*.png')))
    estimate_flow_file_path = osp.join(opt['estimate_flow_root'], f'{video_name}.txt')
    if osp.exists(estimate_flow_file_path):
        with open(estimate_flow_file_path, 'r') as f:
            lines = f.readlines()
            length = len(lines)
        if length == len(frames_path):
            print(f'skip {video_name} since {length}/{len(frames_path)} have done.')
            return
        else:
            print(f're-run {video_name} since only {length}/{len(frames_path)} have done.')

    spynet = SpyNet(load_path=opt['spy_pretrain_weight']).eval().to(device)
    downscale_factor = opt['downscale_factor']

    flow_out_list = []

    pbar = tqdm(total=len(frames_path), unit='frame', desc='worker3')
    pre_img = None
    for idx, frame_path in enumerate(frames_path):
        img_name = osp.basename(frame_path)
        cur_img = read_img(frame_path, device, downscale_factor=downscale_factor)

        if pre_img is not None:
            flow = spynet(cur_img, pre_img)
            flow = flow.abs()
            flow_max = flow.max().item()
            flow_avg = flow.mean().item() * 2.0  # according to lijian's hyper-parameter
        elif idx == 0:
            flow_max = 0.0
            flow_avg = 0.0
        else:
            raise RuntimeError(f'pre_img is none at {idx}')

        flow_out_list.append(f'{img_name} {flow_max:.6f} {flow_avg:.6f}\n')
        pre_img = cur_img

        pbar.update(1)

    with open(estimate_flow_file_path, 'w') as f:
        for line in flow_out_list:
            f.write(line)


# -------------------------------------------------------------------- #
# --------------------------- step4 ---------------------------------- #
# -------------------------------------------------------------------- #


def run_step4(opt):
    black_flag_root = opt['black_flag_root']
    meta_root = opt['meta_files_root']
    os.makedirs(black_flag_root, exist_ok=True)
    if not osp.exists(meta_root):
        print('no videos has run step1, exit.')
        return

    # get the video which has been extracted frames
    videos_name = sorted(glob.glob(osp.join(meta_root, '*.txt')))
    videos_name = [osp.splitext(osp.basename(video_name))[0] for video_name in videos_name]

    if opt['debug']:
        videos_name = videos_name[:3]
        os.makedirs('tmp_black', exist_ok=True)
    else:
        videos_name = videos_name[opt['ss_idx']:opt['to_idx']]

    pbar = tqdm(total=len(videos_name), unit='video', desc='step4')

    pool = Pool(opt['n_thread'])
    for idx, video_name in enumerate(videos_name):
        pool.apply_async(worker4, args=(opt, video_name), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()


def worker4(opt, video_name):
    video_frame_path = osp.join(opt['save_frames_root'], video_name)
    black_flag_path = osp.join(opt['black_flag_root'], f'{video_name}.txt')
    if osp.exists(black_flag_path):
        print(f'skip {video_name} since {black_flag_path} already exists.')
        return

    frames_path = sorted(glob.glob(osp.join(video_frame_path, '*.png')))
    out_list = []
    pbar = tqdm(total=len(frames_path), unit='frame', desc='worker4')

    for frame_path in frames_path:
        img = cv2.imread(frame_path)
        img_name = osp.basename(frame_path)
        h, w = img.shape[0:2]
        total_pixels = h * w
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([img_gray], [0], None, [256], [0.0, 255.0])
        max_pixel = max(hist)[0]
        percentage = max_pixel / total_pixels

        if percentage > opt['black_threshold']:
            out_list.append(f'{img_name} {0} {percentage:.6f}\n')
            if opt['debug']:
                cv2.imwrite(osp.join('tmp_black', f'{video_name}_{img_name}'), img)
        else:
            out_list.append(f'{img_name} {1} {percentage:.6f}\n')

        pbar.update(1)

    with open(black_flag_path, 'w') as f:
        for line in out_list:
            f.write(line)


# -------------------------------------------------------------------- #
# --------------------------- step5 ---------------------------------- #
# -------------------------------------------------------------------- #


def run_step5(opt):
    iqa_score_root = opt['iqa_score_root']
    meta_root = opt['meta_files_root']
    os.makedirs(iqa_score_root, exist_ok=True)
    if not osp.exists(meta_root):
        print('no videos has run step1, exit.')
        return

    # get the video which has been extracted frames
    videos_name = sorted(glob.glob(osp.join(meta_root, '*.txt')))
    videos_name = [osp.splitext(osp.basename(video_name))[0] for video_name in videos_name]

    if opt['debug']:
        videos_name = videos_name[:3]
        os.makedirs('tmp_low_iqa', exist_ok=True)
    else:
        videos_name = videos_name[opt['ss_idx']:opt['to_idx']]

    pbar = tqdm(total=len(videos_name), unit='video', desc='step5')

    num_gpus = torch.cuda.device_count()
    ctx = torch.multiprocessing.get_context('spawn')
    pool = ctx.Pool(min(3 * num_gpus, opt['n_thread']))
    for idx, video_name in enumerate(videos_name):
        pool.apply_async(
            worker5, args=(opt, video_name, torch.device(idx % num_gpus)), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()


@torch.no_grad()
def worker5(opt, video_name, device):
    video_frame_path = osp.join(opt['save_frames_root'], video_name)
    frames_path = sorted(glob.glob(osp.join(video_frame_path, '*.png')))
    iqa_score_path = osp.join(opt['iqa_score_root'], f'{video_name}.txt')
    if osp.exists(iqa_score_path):
        with open(iqa_score_path, 'r') as f:
            lines = f.readlines()
            length = len(lines)
        if length == len(frames_path):
            print(f'skip {video_name} since {length}/{len(frames_path)} have done.')
            return
        else:
            print(f're-run {video_name} since only {length}/{len(frames_path)} have done.')

    assess_net = init_assessment_model('hypernet', device=device)
    assess_net = assess_net.half()

    # specified transformation in original hyperIQA
    transforms_resize = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 384)),
    ])
    transforms_crop = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    iqa_out_list = []

    pbar = tqdm(total=len(frames_path), unit='frame', desc='worker3')
    for idx, frame_path in enumerate(frames_path):
        img_name = osp.basename(frame_path)
        cv2_img = cv2.imread(frame_path)
        # BRG -> RGB
        img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        patchs = []
        img_resize = transforms_resize(img)
        for _ in range(opt['num_patch_per_iqa']):
            patchs.append(transforms_crop(img_resize))
        patch = torch.stack(patchs, dim=0).to(device)

        pred = assess_net(patch.half())
        score = pred.mean().item()

        iqa_out_list.append(f'{img_name} {score:.6f}\n')
        if opt['debug'] and score < 50.0:
            cv2.imwrite(osp.join('tmp_low_iqa', f'{video_name}_{img_name}'), cv2_img)

        pbar.update(1)

    with open(iqa_score_path, 'w') as f:
        for line in iqa_out_list:
            f.write(line)


# -------------------------------------------------------------------- #
# --------------------------- step6 ---------------------------------- #
# -------------------------------------------------------------------- #


def filter_frozen_shots(shots, flows):
    """select clips from input video."""
    flag_shot = np.ones(len(shots))

    for idx, shot in enumerate(shots):
        shot = shot.split(' ')
        start = int(shot[0])
        end = int(shot[1])

        flow_in_shot = []

        for i in range(start, end + 1, 1):
            if i == 0:
                continue
            else:
                flow_in_shot.append(float(flows[i].split(' ')[2]))

        flow_in_shot = np.array(flow_in_shot)
        flow_std = np.std(flow_in_shot)

        if flow_std < 14.0:
            flag_shot[idx] = 0

    return flag_shot


def generate_clips(shots, flows, filter_frames, hyperiqa, max_length=500):
    """
    hyperiqa [0, 100]
    flows [0, 15000] (may be larger)
    """
    clips = []
    clip_scores = []
    clip = []
    shot_flow = 0
    shot_hyperiqa = 0
    for shot in shots:
        shot = shot.split(' ')
        start = int(shot[0])
        end = int(shot[1])

        pre_black = 0
        for i in range(start, end + 1, 1):
            if i == start:
                stat = 0
                pre_black = 1  # the first frame in shot do not need flow
            else:
                stat = 1

            black_frame_thr = float(filter_frames[i].split(' ')[2])

            # drop img when 90% of pixels are identical
            if black_frame_thr < 0.90:
                black_frame = 0
            else:
                black_frame = 1

            # if current frame is a black frame, delete
            if black_frame == 1:
                pre_black = 1
            elif pre_black == 0:
                flow = float(flows[i].split(' ')[1])
                shot_flow += flow
            else:
                pre_black = 0
                flow = float(flows[i].split(' ')[1])

            # calcu hyperiqa for non-black frames
            if black_frame == 0:
                curr_hyperiqa = float(hyperiqa[i].split(' ')[1])
                shot_hyperiqa += curr_hyperiqa

                clip.append(f'{i+1:08d} {stat} {flow} {curr_hyperiqa}')

                if len(clip) == max_length:
                    clips.append(clip.copy())
                    clip_score = shot_flow / 150.0 + shot_hyperiqa
                    clip_score = clip_score / len(clip)
                    clip_scores.append(clip_score)
                    clip = []
                    shot_flow = 0
                    shot_hyperiqa = 0

    # print(len(clip))
    # if len(clip) > 0:
    #     clips.append(clip.copy())
    #     clip_score = shot_flow / 150.0 + shot_hyperiqa
    #     clip_score = clip_score / len(clip)
    #     clip_scores.append(clip_score)

    sorted_shot = np.argsort(-np.array(clip_scores))

    return [clips[i] for i in sorted_shot], [clip_scores[i] for i in sorted_shot]


def run_step6(opt):
    meta_root = opt['meta_files_root']
    if not osp.exists(meta_root):
        print('no videos has run step1, exit.')
        return

    # get the video which has been extracted frames
    videos_name = sorted(glob.glob(osp.join(meta_root, '*.txt')))
    videos_name = [osp.splitext(osp.basename(video_name))[0] for video_name in videos_name]

    if opt['debug']:
        videos_name = videos_name[:3]
    else:
        videos_name = videos_name[opt['ss_idx']:opt['to_idx']]

    pbar = tqdm(total=len(videos_name), unit='video', desc='step6')

    os.makedirs(opt['select_clips_meta'], exist_ok=True)
    os.makedirs(opt['select_clips_frames'], exist_ok=True)
    os.makedirs(opt['select_done_flags'], exist_ok=True)

    pool = Pool(opt['n_thread'])
    for video_name in videos_name:
        pool.apply_async(worker6, args=(opt, video_name), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()


def worker6(opt, video_name):
    select_clips_meta = opt['select_clips_meta']
    select_clips_frames = opt['select_clips_frames']
    select_done_flags = opt['select_done_flags']

    if osp.exists(osp.join(select_done_flags, f'{video_name}.txt')):
        print(f'skip {video_name}.')
        return

    with open(osp.join(opt['detect_shot_root'], f'{video_name}.txt'), 'r') as f:
        shots = f.readlines()
        shots = [shot.strip() for shot in shots]
    with open(osp.join(opt['estimate_flow_root'], f'{video_name}.txt'), 'r') as f:
        flows = f.readlines()
        flows = [flow.strip() for flow in flows]
    with open(osp.join(opt['black_flag_root'], f'{video_name}.txt'), 'r') as f:
        black_flags = f.readlines()
        black_flags = [black_flag.strip() for black_flag in black_flags]
    with open(osp.join(opt['iqa_score_root'], f'{video_name}.txt'), 'r') as f:
        iqa_scores = f.readlines()
        iqa_scores = [iqa_score.strip() for iqa_score in iqa_scores]

    flag_shot = filter_frozen_shots(shots, flows)
    flag = np.where(flag_shot == 1)
    flag = flag[0].tolist()
    filtered_shots = [shots[i] for i in flag]

    clips, scores = generate_clips(
        filtered_shots, flows, black_flags, iqa_scores, max_length=opt['num_frames_per_clip'])
    with open(osp.join(select_clips_meta, f'{video_name}.txt'), 'w') as f:
        for i, clip in enumerate(clips):
            os.makedirs(osp.join(select_clips_frames, f'{video_name}_{i}'), exist_ok=True)
            for idx, info in enumerate(clip):
                f.write(f'clip: {i:02d} {info} {scores[i]}\n')
                img_name = info.split(' ')[0] + '.png'
                shutil.copy(
                    osp.join(opt['save_frames_root'], video_name, img_name),
                    osp.join(select_clips_frames, f'{video_name}_{i}', f'{idx:08d}.png'))
            if i >= opt['num_clips_per_video'] - 1:
                break

    with open(osp.join(select_done_flags, f'{video_name}.txt'), 'w') as f:
        f.write(f'{i+1} clips are selected for {video_name}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataroot',
        type=str,
        required=True,
        help='dataset root, dataroot/raw_videos should contains your HQ videos to be processed.')
    parser.add_argument('--n_thread', type=int, default=4, help='Thread number.')
    parser.add_argument('--run', type=str, default='123456', help='run which steps')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ss_idx', type=int, default=None, help='ss index')
    parser.add_argument('--to_idx', type=int, default=None, help='to index')
    parser.add_argument('--n_frames_per_clip', type=int, default=100)
    parser.add_argument('--n_clips_per_video', type=int, default=1)
    parser.add_argument('--select_clip_root', type=str, default='select_clips')
    args = parser.parse_args()

    main(args)
