import cv2
import os
import torch
from collections import OrderedDict
from os import path as osp
from torch import distributed as dist
from tqdm import tqdm

from basicsr.models.video_base_model import VideoBaseModel
from basicsr.utils import USMSharp, get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class VideoRecurrentCustomModel(VideoBaseModel):

    def __init__(self, opt):
        super(VideoRecurrentCustomModel, self).__init__(opt)
        if self.is_train:
            self.fix_flow_iter = opt['train'].get('fix_flow')
        self.idx = 1
        lq_from_usm = opt['datasets']['train'].get('lq_from_usm', False)
        assert lq_from_usm is False
        self.usm_sharp_gt = opt['datasets']['train'].get('usm_sharp_gt', False)

        if self.usm_sharp_gt:
            usm_radius = opt['datasets']['train'].get('usm_radius', 50)
            self.usm_sharpener = USMSharp(radius=usm_radius).cuda()
            self.usm_weight = opt['datasets']['train'].get('usm_weight', 0.5)
            self.usm_threshold = opt['datasets']['train'].get('usm_threshold', 10)

    @torch.no_grad()
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            if 'gt_usm' in data:
                self.gt_usm = data['gt_usm'].to(self.device)
                logger = get_root_logger()
                logger.warning(
                    'since lq is not from gt_usm, '
                    'we should put the usm_sharp operation outside the dataloader to speed up the traning time')
            elif self.usm_sharp_gt:
                b, n, c, h, w = self.gt.size()
                self.gt_usm = self.usm_sharpener(
                    self.gt.view(b * n, c, h, w), weight=self.usm_weight,
                    threshold=self.usm_threshold).view(b, n, c, h, w)

        # if self.opt['rank'] == 0 and 'debug' in self.opt['name']:
        #     import torchvision
        #     os.makedirs('tmp/gt', exist_ok=True)
        #     os.makedirs('tmp/gt_usm', exist_ok=True)
        #     os.makedirs('tmp/lq', exist_ok=True)
        #     print(self.idx)
        #     for i in range(15):
        #         torchvision.utils.save_image(
        #             self.lq[:, i, :, :, :],
        #             f'tmp/lq/lq{self.idx}_{i}.png',
        #             nrow=4,
        #             padding=2,
        #             normalize=True,
        #             range=(0, 1))
        #         torchvision.utils.save_image(
        #             self.gt[:, i, :, :, :],
        #             f'tmp/gt/gt{self.idx}_{i}.png',
        #             nrow=4,
        #             padding=2,
        #             normalize=True,
        #             range=(0, 1))
        #         torchvision.utils.save_image(
        #             self.gt_usm[:, i, :, :, :],
        #             f'tmp/gt_usm/gt_usm{self.idx}_{i}.png',
        #             nrow=4,
        #             padding=2,
        #             normalize=True,
        #             range=(0, 1))
        #     self.idx += 1
        #     if self.idx >= 20:
        #         exit()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate flow params and normal params for different lr
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters_base(self, current_iter):
        if self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'edvr' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

    def optimize_parameters(self, current_iter):
        self.optimize_parameters_base(current_iter)
        _, _, c, h, w = self.output.size()

        pix_gt = self.gt
        percep_gt = self.gt
        if self.opt.get('l1_gt_usm', False):
            pix_gt = self.gt_usm
        if self.opt.get('percep_gt_usm', False):
            percep_gt = self.gt_usm

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, pix_gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output.view(-1, c, h, w), percep_gt.view(-1, c, h, w))
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """dist_test actually, no gt, no metrics"""
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        assert dataset_name.endswith('CoreFrames')
        rank, world_size = get_dist_info()

        num_folders = len(dataset)
        num_pad = (world_size - (num_folders % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='folder')
            os.makedirs(osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter)), exist_ok=True)

        if self.opt['dist']:
            dist.barrier()
        # Will evaluate (num_folders + num_pad) times, but only the first num_folders results will be recorded.
        # (To avoid wait-dead)
        for i in range(rank, num_folders + num_pad, world_size):
            idx = min(i, num_folders - 1)
            val_data = dataset[idx]
            folder = val_data['folder']

            # compute outputs
            val_data['lq'].unsqueeze_(0)
            self.feed_data(val_data)
            val_data['lq'].squeeze_(0)

            self.test()
            visuals = self.get_current_visuals()

            # tentative for out of GPU memory
            del self.lq
            del self.output
            if 'gt' in visuals:
                del self.gt
            torch.cuda.empty_cache()

            # evaluate
            if i < num_folders:
                for idx in range(visuals['result'].size(1)):
                    result = visuals['result'][0, idx, :, :, :]
                    result_img = tensor2img([result])  # uint8, bgr

                    # since we keep all frames, scale of 4 is not very friendly to storage space
                    # so we use a default scale of 2 to save the frames
                    save_scale = self.opt.get('savescale', 2)
                    net_scale = self.opt.get('scale')
                    if save_scale != net_scale:
                        h, w = result_img.shape[0:2]
                        result_img = cv2.resize(
                            result_img, (w // net_scale * save_scale, h // net_scale * save_scale),
                            interpolation=cv2.INTER_LANCZOS4)

                    if save_img:
                        img_path = osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter),
                                            f"{folder}_{idx:08d}_{self.opt['name'][:5]}.png")
                        # image name only for REDS dataset
                        imwrite(result_img, img_path)

                # progress bar
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Folder: {folder}')

        if rank == 0:
            pbar.close()

    def test(self):
        n = self.lq.size(1)
        self.net_g.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)

        if flip_seq:
            self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)

        with torch.no_grad():
            self.output = self.net_g(self.lq)

        if flip_seq:
            output_1 = self.output[:, :n, :, :, :]
            output_2 = self.output[:, n:, :, :, :].flip(1)
            self.output = 0.5 * (output_1 + output_2)

        if self.center_frame_only:
            self.output = self.output[:, n // 2, :, :, :]

        self.net_g.train()
