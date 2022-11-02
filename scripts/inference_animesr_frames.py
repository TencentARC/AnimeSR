"""inference AnimeSR on frames"""
import argparse
import cv2
import glob
import numpy as np
import os
import psutil
import queue
import threading
import time
import torch
from os import path as osp
from tqdm import tqdm

from animesr.utils.inference_base import get_base_argument_parser, get_inference_model
from animesr.utils.video_util import frames2video
from basicsr.data.transforms import mod_crop
from basicsr.utils.img_util import img2tensor, tensor2img


def read_img(path, require_mod_crop=True, mod_scale=4, input_rescaling_factor=1.0):
    """ read an image tensor from a given path
    Args:
        path: image path
        require_mod_crop: mod crop or not. since the arch is multi-scale, so mod crop is needed by default
        mod_scale: scale factor for mod_crop


    Returns:
        torch.Tensor: size(1, c, h, w)
    """
    img = cv2.imread(path)
    img = img.astype(np.float32) / 255.

    if input_rescaling_factor != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(
            img, (int(w * input_rescaling_factor), int(h * input_rescaling_factor)), interpolation=cv2.INTER_LANCZOS4)

    if require_mod_crop:
        img = mod_crop(img, mod_scale)

    img = img2tensor(img, bgr2rgb=True, float32=True)
    return img.unsqueeze(0)


class IOConsumer(threading.Thread):
    """Since IO time can take up a significant portion of the total inference time,
    so we use multi thread to write frames individually.
    """

    def __init__(self, args: argparse.Namespace, que, qid):
        super().__init__()
        self._queue = que
        self.qid = qid
        self.args = args

    def run(self):
        while True:
            msg = self._queue.get()
            if isinstance(msg, str) and msg == 'quit':
                break

            output = msg['output']
            imgname = msg['imgname']
            out_img = tensor2img(output.squeeze(0))
            if self.args.outscale != self.args.netscale:
                h, w = out_img.shape[:2]
                out_img = cv2.resize(
                    out_img, (int(
                        w * self.args.outscale / self.args.netscale), int(h * self.args.outscale / self.args.netscale)),
                    interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(imgname, out_img)

        print(f'IO for worker {self.qid} is done.')


@torch.no_grad()
def main():
    """Inference demo for AnimeSR.
    It mainly for restoring anime frames.
    """
    parser = get_base_argument_parser()
    parser.add_argument('--input_rescaling_factor', type=float, default=1.0)
    parser.add_argument('--num_io_consumer', type=int, default=3, help='number of IO consumer')
    parser.add_argument(
        '--sample_interval',
        type=int,
        default=1,
        help='save 1 frame for every $sample_interval frames. this will be useful for calculating the metrics')
    parser.add_argument('--save_video_too', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_inference_model(args, device)

    # prepare output dir
    frame_output = osp.join(args.output, args.expname, 'frames')
    os.makedirs(frame_output, exist_ok=True)

    # the input format can be:
    # 1. clip folder which contains frames
    # or 2. a folder which contains several clips
    first_level_dir = len(glob.glob(osp.join(args.input, '*.png'))) > 0
    if args.input.endswith('/'):
        args.input = args.input[:-1]
    if first_level_dir:
        videos_name = [osp.basename(args.input)]
        args.input = osp.dirname(args.input)
    else:
        videos_name = sorted(os.listdir(args.input))

    pbar1 = tqdm(total=len(videos_name), unit='video', desc='inference')

    que = queue.Queue()
    consumers = [IOConsumer(args, que, f'IO_{i}') for i in range(args.num_io_consumer)]
    for consumer in consumers:
        consumer.start()

    for video_name in videos_name:
        video_folder_path = osp.join(args.input, video_name)
        imgs_list = sorted(glob.glob(osp.join(video_folder_path, '*')))
        num_imgs = len(imgs_list)
        os.makedirs(osp.join(frame_output, video_name), exist_ok=True)

        # prepare
        prev = read_img(
            imgs_list[0],
            require_mod_crop=True,
            mod_scale=args.mod_scale,
            input_rescaling_factor=args.input_rescaling_factor).to(device)
        cur = prev
        nxt = read_img(
            imgs_list[min(1, num_imgs - 1)],
            require_mod_crop=True,
            mod_scale=args.mod_scale,
            input_rescaling_factor=args.input_rescaling_factor).to(device)
        c, h, w = prev.size()[-3:]
        state = prev.new_zeros(1, 64, h, w)
        out = prev.new_zeros(1, c, h * args.netscale, w * args.netscale)

        pbar2 = tqdm(total=num_imgs, unit='frame', desc='inference')
        tot_model_time = 0
        cnt_model_time = 0
        for idx in range(num_imgs):
            torch.cuda.synchronize()
            start = time.time()
            img_name = osp.splitext(osp.basename(imgs_list[idx]))[0]

            out, state = model.cell(torch.cat((prev, cur, nxt), dim=1), out, state)

            torch.cuda.synchronize()
            model_time = time.time() - start
            tot_model_time += model_time
            cnt_model_time += 1

            if (idx + 1) % args.sample_interval == 0:
                # put the output frame to the queue to be consumed
                que.put({'output': out.cpu().clone(), 'imgname': osp.join(frame_output, video_name, f'{img_name}.png')})

            torch.cuda.synchronize()
            start = time.time()
            prev = cur
            cur = nxt
            nxt = read_img(
                imgs_list[min(idx + 2, num_imgs - 1)],
                require_mod_crop=True,
                mod_scale=args.mod_scale,
                input_rescaling_factor=args.input_rescaling_factor).to(device)
            torch.cuda.synchronize()
            read_time = time.time() - start

            pbar2.update(1)
            pbar2.set_description(f'read_time: {read_time}, model_time: {tot_model_time/cnt_model_time}')

            mem = psutil.virtual_memory()
            # since the speed of producer (model inference) is faster than the consumer (I/O)
            # if there is a risk of OOM, just sleep to let the consumer work
            if mem.percent > 80.0:
                time.sleep(30)

        pbar1.update(1)

    for _ in range(args.num_io_consumer):
        que.put('quit')
    for consumer in consumers:
        consumer.join()

    if not args.save_video_too:
        return

    # convert the frames to videos
    video_output = osp.join(args.output, args.expname, 'videos')
    os.makedirs(video_output, exist_ok=True)
    for video_name in videos_name:
        out_path = osp.join(video_output, f'{video_name}.mp4')
        frames2video(
            osp.join(frame_output, video_name), out_path, fps=24 if args.fps is None else args.fps, suffix='png')


if __name__ == '__main__':
    main()
