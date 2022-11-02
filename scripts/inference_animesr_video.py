import cv2
import ffmpeg
import glob
import mimetypes
import numpy as np
import os
import shutil
import subprocess
import torch
from os import path as osp
from tqdm import tqdm

from animesr.utils import video_util
from animesr.utils.inference_base import get_base_argument_parser, get_inference_model
from basicsr.data.transforms import mod_crop
from basicsr.utils.img_util import img2tensor, tensor2img
from basicsr.utils.logger import AvgTimer


def get_video_meta_info(video_path):
    """get the meta info of the video by using ffprobe with python interface"""
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    try:
        ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    except KeyError:  # bilibili transcoder dont have nb_frames
        ret['duration'] = float(probe['format']['duration'])
        ret['nb_frames'] = int(ret['duration'] * ret['fps'])
        print(ret['duration'], ret['nb_frames'])
    return ret


def get_sub_video(args, num_process, process_idx):
    """Cut the whole video into num_process parts, return the process_idx-th part"""
    if num_process == 1:
        return args.input
    meta = get_video_meta_info(args.input)
    duration = int(meta['nb_frames'] / meta['fps'])
    part_time = duration // num_process
    print(f'duration: {duration}, part_time: {part_time}')
    out_path = osp.join(args.output, 'inp_sub_videos', f'{process_idx:03d}.mp4')
    cmd = [
        args.ffmpeg_bin,
        f'-i {args.input}',
        f'-ss {part_time * process_idx}',
        f'-to {part_time * (process_idx + 1)}' if process_idx != num_process - 1 else '',
        '-async 1',
        out_path,
        '-y',
    ]
    print(' '.join(cmd))
    subprocess.call(' '.join(cmd), shell=True)
    return out_path


class Reader:
    """read frames from a video stream or frames list"""

    def __init__(self, args, total_workers=1, worker_idx=0, device=torch.device('cuda')):
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type is None else input_type
        self.paths = []  # for image&folder type
        self.audio = None
        self.input_fps = None
        if self.input_type.startswith('video'):
            video_path = get_sub_video(args, total_workers, worker_idx)
            # read bgr from stream, which is the same format as opencv
            self.stream_reader = (
                ffmpeg
                .input(video_path)
                .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='error')
                .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)
            )  # yapf: disable  # noqa
            meta = get_video_meta_info(video_path)
            self.width = meta['width']
            self.height = meta['height']
            self.input_fps = meta['fps']
            self.audio = meta['audio']
            self.nb_frames = meta['nb_frames']

        else:
            if self.input_type.startswith('image'):
                self.paths = [args.input]
            else:
                paths = sorted(glob.glob(os.path.join(args.input, '*')))
                tot_frames = len(paths)
                num_frame_per_worker = tot_frames // total_workers + (1 if tot_frames % total_workers else 0)
                self.paths = paths[num_frame_per_worker * worker_idx:num_frame_per_worker * (worker_idx + 1)]

            self.nb_frames = len(self.paths)
            assert self.nb_frames > 0, 'empty folder'
            from PIL import Image
            tmp_img = Image.open(self.paths[0])  # lazy load
            self.width, self.height = tmp_img.size
        self.idx = 0
        self.device = device

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        """the fps of sr video is set to the user input fps first, followed by the input fps,
        If the first two values are None, then the commonly used fps 24 is set"""
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        """return the number of frames for this worker, however, this may be not accurate for video stream"""
        return self.nb_frames

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            # end of stream
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def get_frame_from_list(self):
        if self.idx >= self.nb_frames:
            return None
        img = cv2.imread(self.paths[self.idx])
        self.idx += 1
        return img

    def get_frame(self):
        if self.input_type.startswith('video'):
            img = self.get_frame_from_stream()
        else:
            img = self.get_frame_from_list()

        if img is None:
            raise StopIteration

        # bgr uint8 numpy -> rgb float32 [0, 1] tensor on device
        img = img.astype(np.float32) / 255.
        img = mod_crop(img, self.args.mod_scale)
        img = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0).to(self.device)
        if self.args.half:
            # half precision won't make a big impact on visuals
            img = img.half()
        return img

    def close(self):
        # close the video stream
        if self.input_type.startswith('video'):
            self.stream_reader.stdin.close()
            self.stream_reader.wait()


class Writer:
    """write frames to a video stream"""

    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')

        vsp = video_save_path
        if audio is not None:
            self.stream_writer = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{out_width}x{out_height}', framerate=fps)
                .output(audio, vsp, pix_fmt='yuv420p', vcodec='libx264', loglevel='error', acodec='copy')
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)
            )  # yapf: disable  # noqa
        else:
            self.stream_writer = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{out_width}x{out_height}', framerate=fps)
                .output(vsp, pix_fmt='yuv420p', vcodec='libx264', loglevel='error')
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)
            )  # yapf: disable  # noqa

        self.out_width = out_width
        self.out_height = out_height
        self.args = args

    def write_frame(self, frame):
        if self.args.outscale != self.args.netscale:
            frame = cv2.resize(frame, (self.out_width, self.out_height), interpolation=cv2.INTER_LANCZOS4)
        self.stream_writer.stdin.write(frame.tobytes())

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()


@torch.no_grad()
def inference_video(args, video_save_path, device=None, total_workers=1, worker_idx=0):
    # prepare model
    model = get_inference_model(args, device)

    # prepare reader and writer
    reader = Reader(args, total_workers, worker_idx, device=device)
    audio = reader.get_audio()
    height, width = reader.get_resolution()
    height = height - height % args.mod_scale
    width = width - width % args.mod_scale
    fps = reader.get_fps()
    writer = Writer(args, audio, height, width, video_save_path, fps)

    # initialize pre/cur/nxt frames, pre sr frame, and pre hidden state for inference
    end_flag = False
    prev = reader.get_frame()
    cur = prev
    try:
        nxt = reader.get_frame()
    except StopIteration:
        end_flag = True
        nxt = cur
    state = prev.new_zeros(1, 64, height, width)
    out = prev.new_zeros(1, 3, height * args.netscale, width * args.netscale)

    pbar = tqdm(total=len(reader), unit='frame', desc='inference')
    model_timer = AvgTimer()  # model inference time tracker
    i_timer = AvgTimer()  # I(input read) time tracker
    o_timer = AvgTimer()  # O(output write) time tracker
    while True:
        # inference at current step
        torch.cuda.synchronize(device=device)
        model_timer.start()
        out, state = model.cell(torch.cat((prev, cur, nxt), dim=1), out, state)
        torch.cuda.synchronize(device=device)
        model_timer.record()

        # write current sr frame to video stream
        torch.cuda.synchronize(device=device)
        o_timer.start()
        output_frame = tensor2img(out, rgb2bgr=False)
        writer.write_frame(output_frame)
        torch.cuda.synchronize(device=device)
        o_timer.record()

        # if end of stream, break
        if end_flag:
            break

        # move the sliding window
        torch.cuda.synchronize(device=device)
        i_timer.start()
        prev = cur
        cur = nxt
        try:
            nxt = reader.get_frame()
        except StopIteration:
            nxt = cur
            end_flag = True
        torch.cuda.synchronize(device=device)
        i_timer.record()

        # update&print infomation
        pbar.update(1)
        pbar.set_description(
            f'I: {i_timer.get_avg_time():.4f} O: {o_timer.get_avg_time():.4f} Model: {model_timer.get_avg_time():.4f}')

    reader.close()
    writer.close()


def run(args):
    if args.suffix is None:
        args.suffix = ''
    else:
        args.suffix = f'_{args.suffix}'
    video_save_path = osp.join(args.output, f'{args.video_name}{args.suffix}.mp4')

    # set up multiprocessing
    num_gpus = torch.cuda.device_count()
    num_process = num_gpus * args.num_process_per_gpu
    if num_process == 1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inference_video(args, video_save_path, device=device)
        return

    ctx = torch.multiprocessing.get_context('spawn')
    pool = ctx.Pool(num_process)
    out_sub_videos_dir = osp.join(args.output, 'out_sub_videos')
    os.makedirs(out_sub_videos_dir, exist_ok=True)
    os.makedirs(osp.join(args.output, 'inp_sub_videos'), exist_ok=True)

    pbar = tqdm(total=num_process, unit='sub_video', desc='inference')
    for i in range(num_process):
        sub_video_save_path = osp.join(out_sub_videos_dir, f'{i:03d}.mp4')
        pool.apply_async(
            inference_video,
            args=(args, sub_video_save_path, torch.device(i % num_gpus), num_process, i),
            callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()

    # combine sub videos
    # prepare vidlist.txt
    with open(f'{args.output}/vidlist.txt', 'w') as f:
        for i in range(num_process):
            f.write(f'file \'out_sub_videos/{i:03d}.mp4\'\n')
    # To avoid video&audio desync as mentioned in https://github.com/xinntao/Real-ESRGAN/issues/388
    # we use the solution provided in https://stackoverflow.com/a/52156277 to solve this issue
    cmd = [
        args.ffmpeg_bin,
        '-f', 'concat',
        '-safe', '0',
        '-i', f'{args.output}/vidlist.txt',
        '-c:v', 'copy',
        '-af', 'aresample=async=1000',
        video_save_path,
        '-y',
    ]  # yapf: disable
    print(' '.join(cmd))
    subprocess.call(cmd)
    shutil.rmtree(out_sub_videos_dir)
    shutil.rmtree(osp.join(args.output, 'inp_sub_videos'))
    os.remove(f'{args.output}/vidlist.txt')


def main():
    """Inference demo for AnimeSR.
    It mainly for restoring anime videos.
    """
    parser = get_base_argument_parser()
    parser.add_argument(
        '--extract_frame_first',
        action='store_true',
        help='if input is a video, you can still extract the frames first, other wise AnimeSR will read from stream')
    parser.add_argument(
        '--num_process_per_gpu', type=int, default=1, help='the total process is number_process_per_gpu * num_gpu')
    parser.add_argument(
        '--suffix', type=str, default=None, help='you can add a suffix string to the sr video name, for example, x2')
    args = parser.parse_args()
    args.ffmpeg_bin = os.environ.get('ffmpeg_exe_path', 'ffmpeg')

    args.input = args.input.rstrip('/').rstrip('\\')

    if mimetypes.guess_type(args.input)[0] is not None and mimetypes.guess_type(args.input)[0].startswith('video'):
        is_video = True
    else:
        is_video = False

    if args.extract_frame_first and not is_video:
        args.extract_frame_first = False

    # prepare input and output
    args.video_name = osp.splitext(osp.basename(args.input))[0]
    args.output = osp.join(args.output, args.expname, 'videos', args.video_name)
    os.makedirs(args.output, exist_ok=True)
    if args.extract_frame_first:
        inp_extracted_frames = osp.join(args.output, 'inp_extracted_frames')
        os.makedirs(inp_extracted_frames, exist_ok=True)
        video_util.video2frames(args.input, inp_extracted_frames, force=True, high_quality=True)
        video_meta = get_video_meta_info(args.input)
        args.fps = video_meta['fps']
        args.input = inp_extracted_frames

    run(args)

    if args.extract_frame_first:
        shutil.rmtree(args.input)


if __name__ == '__main__':
    main()
