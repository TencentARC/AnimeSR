import cv2
import ffmpeg
import glob
import numpy as np
import os
import random
import torch
from os import path as osp
from torch.utils import data as data

from basicsr.data.degradations import random_add_gaussian_noise, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from .data_utils import random_crop


@DATASET_REGISTRY.register()
class FFMPEGAnimeDataset(data.Dataset):
    """Anime datasets with only classic basic operators"""

    def __init__(self, opt):
        super(FFMPEGAnimeDataset, self).__init__()
        self.opt = opt
        self.num_frame = opt['num_frame']
        self.num_half_frames = opt['num_frame'] // 2

        self.keys = []
        self.clip_frames = {}

        self.gt_root = opt['dataroot_gt']

        logger = get_root_logger()

        clip_names = os.listdir(self.gt_root)
        for clip_name in clip_names:
            num_frames = len(glob.glob(osp.join(self.gt_root, clip_name, '*.png')))
            self.keys.extend([f'{clip_name}/{i:08d}' for i in range(num_frames)])
            self.clip_frames[clip_name] = num_frames

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False

        self.iso_blur_range = opt.get('iso_blur_range', [0.2, 4])
        self.aniso_blur_range = opt.get('aniso_blur_range', [0.8, 3])
        self.noise_range = opt.get('noise_range', [0, 10])
        self.crf_range = opt.get('crf_range', [18, 35])
        self.ffmpeg_profile_names = opt.get('ffmpeg_profile_names', ['baseline', 'main', 'high'])
        self.ffmpeg_profile_probs = opt.get('ffmpeg_profile_probs', [0.1, 0.2, 0.7])

        self.scale = opt.get('scale', 4)
        assert self.scale in (2, 4)

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def get_gt_clip(self, index):
        """
        get the GT(hr) clip with self.num_frame frames
        :param index: the index from __getitem__
        :return: a list of images, with numpy(cv2) format
        """
        key = self.keys[index]  # get clip from this key frame (if possible)
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the "interval" of neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        center_frame_idx = int(frame_name)
        start_frame_idx = center_frame_idx - self.num_half_frames * interval
        end_frame_idx = center_frame_idx + self.num_half_frames * interval

        # if the index doesn't satisfy the requirement, resample it
        if (start_frame_idx < 0) or (end_frame_idx >= self.clip_frames[clip_name]):
            center_frame_idx = random.randint(self.num_half_frames * interval,
                                              self.clip_frames[clip_name] - 1 - self.num_half_frames * interval)
            start_frame_idx = center_frame_idx - self.num_half_frames * interval
            end_frame_idx = center_frame_idx + self.num_half_frames * interval

        # determine the neighbor frames
        neighbor_list = list(range(start_frame_idx, end_frame_idx + 1, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring GT frames
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_gt_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_gt_path = osp.join(self.gt_root, clip_name, f'{neighbor:08d}.png')

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # random crop
        img_gts = random_crop(img_gts, self.opt['gt_size'])
        # augmentation
        img_gts = augment(img_gts, self.opt['use_flip'], self.opt['use_rot'])

        return img_gts

    def add_ffmpeg_compression(self, img_lqs, width, height):
        # ffmpeg
        loglevel = 'error'
        format = 'h264'
        fps = random.choices([24, 25, 30, 50, 60], [0.2, 0.2, 0.2, 0.2, 0.2])[0]  # still have problems
        fps = 25
        crf = np.random.uniform(self.crf_range[0], self.crf_range[1])

        try:
            extra_args = dict()
            if format == 'h264':
                vcodec = 'libx264'
                profile = random.choices(self.ffmpeg_profile_names, self.ffmpeg_profile_probs)[0]
                extra_args['profile:v'] = profile

            ffmpeg_img2video = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}',
                             r=fps).filter('fps', fps=fps, round='up').output(
                                 'pipe:', format=format, pix_fmt='yuv420p', crf=crf, vcodec=vcodec,
                                 **extra_args).global_args('-hide_banner').global_args('-loglevel', loglevel).run_async(
                                     pipe_stdin=True, pipe_stdout=True))
            ffmpeg_video2img = (
                ffmpeg.input('pipe:', format=format).output('pipe:', format='rawvideo',
                                                            pix_fmt='rgb24').global_args('-hide_banner').global_args(
                                                                '-loglevel',
                                                                loglevel).run_async(pipe_stdin=True, pipe_stdout=True))

            # read a sequence of images
            for img_lq in img_lqs:
                ffmpeg_img2video.stdin.write(img_lq.astype(np.uint8).tobytes())

            ffmpeg_img2video.stdin.close()
            video_bytes = ffmpeg_img2video.stdout.read()
            ffmpeg_img2video.wait()

            # ffmpeg: video to images
            ffmpeg_video2img.stdin.write(video_bytes)
            ffmpeg_video2img.stdin.close()
            img_lqs_ffmpeg = []
            while True:
                in_bytes = ffmpeg_video2img.stdout.read(width * height * 3)
                if not in_bytes:
                    break
                in_frame = (np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3]))
                in_frame = in_frame.astype(np.float32) / 255.
                img_lqs_ffmpeg.append(in_frame)

            ffmpeg_video2img.wait()

            assert len(img_lqs_ffmpeg) == self.num_frame, 'Wrong length'
        except AssertionError as error:
            logger = get_root_logger()
            logger.warn(f'ffmpeg assertion error: {error}')
        except Exception as error:
            logger = get_root_logger()
            logger.warn(f'ffmpeg exception error: {error}')
        else:
            img_lqs = img_lqs_ffmpeg

        return img_lqs

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        img_gts = self.get_gt_clip(index)

        # ------------- generate LQ frames --------------#
        # add blur
        kernel = random_mixed_kernels(['iso', 'aniso'], [0.7, 0.3], 21, self.iso_blur_range, self.aniso_blur_range)
        img_lqs = [cv2.filter2D(v, -1, kernel) for v in img_gts]
        # add noise
        img_lqs = [
            random_add_gaussian_noise(v, sigma_range=self.noise_range, gray_prob=0.5, clip=True, rounds=False)
            for v in img_lqs
        ]
        # downsample
        h, w = img_gts[0].shape[0:2]
        width = w // self.scale
        height = h // self.scale
        resize_type = random.choices([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC], [0.3, 0.3, 0.4])[0]
        img_lqs = [cv2.resize(v, (width, height), interpolation=resize_type) for v in img_lqs]
        # ffmpeg
        img_lqs = [np.clip(img_lq * 255.0, 0, 255) for img_lq in img_lqs]
        img_lqs = self.add_ffmpeg_compression(img_lqs, width, height)
        # ------------- end --------------#
        img_gts = img2tensor(img_gts)
        img_lqs = img2tensor(img_lqs)
        img_gts = torch.stack(img_gts, dim=0)
        img_lqs = torch.stack(img_lqs, dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        return {'lq': img_lqs, 'gt': img_gts}

    def __len__(self):
        return len(self.keys)
