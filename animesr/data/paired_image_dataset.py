import glob
import os
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment, mod_crop, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CustomPairedImageDataset(data.Dataset):
    """Paired image dataset for training LBO.

    Read real-world LQ and GT frames pairs.
    The organization of these gt&lq folder is similar to AVC-Train,
    except that each folder contains 200 clips, and each clip contains 11 frames.
    We will ignore the first frame, so there are finally 2000 training pair data.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt, also the pseudo HR path.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(CustomPairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.mod_crop_scale = 8

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']

        omit_first_frame = opt.get('omit_first_frame', True)
        start_idx = 1 if omit_first_frame else 0

        self.paths = []
        clip_list = os.listdir(self.lq_folder)
        for clip_name in clip_list:
            lq_frame_list = sorted(glob.glob(f'{self.lq_folder}/{clip_name}/*.png'))
            gt_frame_list = sorted(glob.glob(f'{self.gt_folder}/{clip_name}/*.png'))
            assert len(lq_frame_list) == len(gt_frame_list)
            for i in range(start_idx, len(lq_frame_list)):
                # omit the first frame
                self.paths.append(dict([('lq_path', lq_frame_list[i]), ('gt_path', gt_frame_list[i])]))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        img_lq = mod_crop(img_lq, self.mod_crop_scale)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
