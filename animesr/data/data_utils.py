import random
import torch


def random_crop(imgs, patch_size, top=None, left=None):
    """
    randomly crop patches from imgs
    :param imgs: can be (list of) tensor, cv2 img
    :param patch_size: patch size, usually 256
    :param top: will sample if is None
    :param left: will sample if is None
    :return: cropped patches from input imgs
    """
    if not isinstance(imgs, list):
        imgs = [imgs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(imgs[0]) else 'Numpy'

    if input_type == 'Tensor':
        h, w = imgs[0].size()[-2:]
    else:
        h, w = imgs[0].shape[0:2]

    # randomly choose top and left coordinates
    if top is None:
        top = random.randint(0, h - patch_size)
    if left is None:
        left = random.randint(0, w - patch_size)

    if input_type == 'Tensor':
        imgs = [v[:, :, top:top + patch_size, left:left + patch_size] for v in imgs]
    else:
        imgs = [v[top:top + patch_size, left:left + patch_size, ...] for v in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]
    return imgs
