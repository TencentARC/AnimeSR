import numpy as np
import random
import torch
from torch.nn import functional as F

from animesr.archs.simple_degradation_arch import SimpleDegradationArch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_mixed_kernels
from basicsr.utils import FileClient, get_root_logger, img2tensor
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import DATASET_REGISTRY
from .ffmpeg_anime_dataset import FFMPEGAnimeDataset


@DATASET_REGISTRY.register()
class FFMPEGAnimeLBODataset(FFMPEGAnimeDataset):
    """Anime datasets with both classic basic operators and learnable basic operators (LBO)"""

    def __init__(self, opt):
        super(FFMPEGAnimeLBODataset, self).__init__(opt)

        self.rank, self.world_size = get_dist_info()

        self.lbo = SimpleDegradationArch(downscale=2)
        lbo_list = opt['degradation_model_path']
        if not isinstance(lbo_list, list):
            lbo_list = [lbo_list]
        self.lbo_list = lbo_list
        # print(f'degradation model path for {self.rank} {self.world_size}: {degradation_model_path}\n')
        # the real load is at reload_degradation_model function
        self.lbo.load_state_dict(torch.load(self.lbo_list[0], map_location=lambda storage, loc: storage)['params'])
        self.lbo = self.lbo.to(f'cuda:{self.rank}').eval()
        self.lbo_prob = opt.get('lbo_prob', 0.5)

    def reload_degradation_model(self):
        """
        __init__ will be only invoked once for one gpu worker, so if we want to
         have num_worker_dataset * num_gpu degradation model, we must call this func in __getitem__
         ref: https://discuss.pytorch.org/t/what-happened-when-set-num-workers-0-in-dataloader/138515
        """
        degradation_model_path = random.choice(self.lbo_list)
        self.lbo.load_state_dict(
            torch.load(degradation_model_path, map_location=lambda storage, loc: storage)['params'])
        print(f'reload degradation model path for {self.rank} {self.world_size}: {degradation_model_path}\n')
        logger = get_root_logger()
        logger.info(f'reload degradation model path for {self.rank} {self.world_size}: {degradation_model_path}\n')

    @torch.no_grad()
    def custom_resize(self, x, scale=2):
        if random.random() < self.lbo_prob:  # learned degradation model from real-world
            x = self.lbo(x)
        else:  # classic synthetic
            h, w = x.shape[2:]
            width = w // scale
            height = h // scale
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            if mode == 'area':
                align_corners = None
            else:
                align_corners = False
            x = F.interpolate(x, size=(height, width), mode=mode, align_corners=align_corners)

        return x

    @torch.no_grad()
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
            # called only once
            self.reload_degradation_model()

        img_gts = self.get_gt_clip(index)

        # ------------- generate LQ frames --------------#
        # change to CUDA implementation
        img_gts = img2tensor(img_gts)
        img_gts = torch.stack(img_gts, dim=0)
        img_gts = img_gts.to(f'cuda:{self.rank}')
        # add blur
        kernel = random_mixed_kernels(['iso', 'aniso'], [0.7, 0.3], 21, self.iso_blur_range, self.aniso_blur_range)
        with torch.no_grad():
            kernel = torch.FloatTensor(kernel).unsqueeze(0).expand(self.num_frame, 21, 21).to(f'cuda:{self.rank}')
            img_lqs = filter2D(img_gts, kernel)
            # add noise
            img_lqs = random_add_gaussian_noise_pt(
                img_lqs, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=0.5)
            # downsample
            img_lqs = self.custom_resize(img_lqs)
            if self.scale == 4:
                img_lqs = self.custom_resize(img_lqs)
            height, width = img_lqs.shape[2:]
            # back to numpy since ffmpeg compression operate on cpu
            img_lqs = img_lqs.detach().clamp_(0, 1).permute(0, 2, 3, 1) * 255  # B, H, W, C
            img_lqs = img_lqs.type(torch.uint8).cpu().numpy()[:, :, :, ::-1]
            img_lqs = np.split(img_lqs, self.num_frame, axis=0)
            img_lqs = [img_lq[0] for img_lq in img_lqs]

        # ffmpeg
        img_lqs = self.add_ffmpeg_compression(img_lqs, width, height)
        # ------------- end --------------#
        img_lqs = img2tensor(img_lqs)
        img_lqs = torch.stack(img_lqs, dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w) on gpu
        return {'lq': img_lqs, 'gt': img_gts.cpu()}

    def __len__(self):
        return len(self.keys)
