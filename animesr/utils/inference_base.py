import argparse
import os.path
import torch

from animesr.archs.vsr_arch import MSRSWVSR


def get_base_argument_parser() -> argparse.ArgumentParser:
    """get the base argument parser for inference scripts"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='input test image folder or video path')
    parser.add_argument('-o', '--output', type=str, default='results', help='save image/video path')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='AnimeSR_v2',
        help='Model names: AnimeSR_v2 | AnimeSR_v1-PaperModel. Default:AnimeSR_v2')
    parser.add_argument(
        '-s',
        '--outscale',
        type=int,
        default=4,
        help='The netscale is x4, but you can achieve arbitrary output scale (e.g., x2) with the argument outscale'
        'The program will further perform cheap resize operation after the AnimeSR output. '
        'This is useful when you want to save disk space or avoid too large-resolution output')
    parser.add_argument(
        '--expname', type=str, default='animesr', help='A unique name to identify your current inference')
    parser.add_argument(
        '--netscale',
        type=int,
        default=4,
        help='the released models are all x4 models, only change this if you train a x2 or x1 model by yourself')
    parser.add_argument(
        '--mod_scale',
        type=int,
        default=4,
        help='the scale used for mod crop, since AnimeSR use a multi-scale arch, so the edge should be divisible by 4')
    parser.add_argument('--fps', type=int, default=None, help='fps of the sr videos')
    parser.add_argument('--half', action='store_true', help='use half precision to inference')

    return parser


def get_inference_model(args, device) -> MSRSWVSR:
    """return an on device model with eval mode"""
    # set up model
    model = MSRSWVSR(num_feat=64, num_block=[5, 3, 2], netscale=args.netscale)

    model_path = f'weights/{args.model_name}.pth'
    assert os.path.isfile(model_path), \
        f'{model_path} does not exist, please make sure you successfully download the pretrained models ' \
        f'and put them into the weights folder'

    # load checkpoint
    loadnet = torch.load(model_path)
    model.load_state_dict(loadnet, strict=True)
    model.eval()
    model = model.to(device)

    # num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    # print(num_parameters)
    # exit(0)

    return model.half() if args.half else model
