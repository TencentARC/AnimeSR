import glob
import os
import subprocess

default_ffmpeg_exe_path = 'ffmpeg'
default_ffprobe_exe_path = 'ffprobe'
default_ffmpeg_vcodec = 'h264'
default_ffmpeg_pix_fmt = 'yuv420p'


def get_video_fps(video_path, ret_type='float'):
    """Get the fps of the video.

    Args:
        video_path (str): the video path;
        ret_type (str): the return type, it supports `str`, `float`, and `tuple` (numerator, denominator).

    Returns:
        --fps (str): if ret_type is `str`.
        --fps (float): if ret_type is `float`.
        --fps (tuple): if ret_type is tuple, (numerator, denominator).
    """

    global default_ffprobe_exe_path

    ffprobe_exe_path = os.environ.get('ffprobe_exe_path', default_ffprobe_exe_path)

    cmd = [
        ffprobe_exe_path, '-v', 'quiet', '-select_streams', 'v', '-of', 'default=noprint_wrappers=1:nokey=1',
        '-show_entries', 'stream=r_frame_rate', video_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    fps = result.stdout.decode('utf-8').strip()

    # e.g. 30/1
    numerator, denominator = map(lambda x: int(x), fps.split('/'))
    if ret_type == 'float':
        return numerator / denominator
    elif ret_type == 'str':
        return str(numerator / denominator)
    else:
        return numerator, denominator


def get_video_num_frames(video_path):
    """Get the video's total number of frames."""

    global default_ffprobe_exe_path

    ffprobe_exe_path = os.environ.get('ffprobe_exe_path', default_ffprobe_exe_path)

    cmd = [
        ffprobe_exe_path, '-v', 'quiet', '-select_streams', 'v', '-count_packets', '-of',
        'default=noprint_wrappers=1:nokey=1', '-show_entries', 'stream=nb_read_packets', video_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    nb_frames = result.stdout.decode('utf-8').strip()

    return int(nb_frames)


def get_video_bitrate(video_path):
    """Get the bitrate of the video."""

    global default_ffprobe_exe_path

    ffprobe_exe_path = os.environ.get('ffprobe_exe_path', default_ffprobe_exe_path)

    cmd = [
        ffprobe_exe_path, '-v', 'quiet', '-select_streams', 'v', '-of', 'default=noprint_wrappers=1:nokey=1',
        '-show_entries', 'stream=bit_rate', video_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    bitrate = result.stdout.decode('utf-8').strip()

    if bitrate == 'N/A':
        return bitrate

    return int(bitrate) // 1000


def get_video_resolution(video_path):
    """Get the resolution (h and w) of the video.

    Args:
        video_path (str): the video path;

    Returns:
        h, w (int)
    """

    global default_ffprobe_exe_path

    ffprobe_exe_path = os.environ.get('ffprobe_exe_path', default_ffprobe_exe_path)

    cmd = [
        ffprobe_exe_path, '-v', 'quiet', '-select_streams', 'v', '-of', 'csv=s=x:p=0', '-show_entries',
        'stream=width,height', video_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    resolution = result.stdout.decode('utf-8').strip()

    # print(resolution)

    w, h = resolution.split('x')
    return int(h), int(w)


def video2frames(video_path, out_dir, force=False, high_quality=True, ss=None, to=None, vf=None):
    """Extract frames from the video

    Args:
        out_dir: where to save the frames
        force: if out_dir is not empty, forceTrue will still extract frames
        high_quality: whether to use the highest quality
        ss: start time, format HH.MM.SS[.xxx], if None, extract full video
        to: end time, format HH.MM.SS[.xxx], if None, extract full video
        vf: video filter
    """
    global default_ffmpeg_exe_path

    ffmpeg_exc_path = os.environ.get('ffmpeg_exe_path', default_ffmpeg_exe_path)

    imgs = glob.glob(os.path.join(out_dir, '*.png'))
    length = len(imgs)
    if length > 0:
        print(f'{out_dir} already has frames!, force extract = {force}')
        if not force:
            return out_dir

    print(f'extracting frames for {video_path}')

    cmd = [
        ffmpeg_exc_path,
        f'-i {video_path}',
        '-v error',
        f'-ss {ss} -to {to}' if ss is not None and to is not None else '',
        '-qscale:v 1 -qmin 1 -qmax 1 -vsync 0' if high_quality else '',
        f'-vf {vf}' if vf is not None else '',
        f'{out_dir}/%08d.png',
    ]
    print(' '.join(cmd))
    subprocess.call(' '.join(cmd), shell=True)

    return out_dir


def frames2video(frames_dir, out_path, fps=25, filter='*', suffix=None):
    """Combine frames under a folder to video
    Args:
        frames_dir: input folder where frames locate
        out_path: the output video path
        fps: output video fps
        suffix: the frame suffix, e.g., png jpg
    """
    global default_ffmpeg_vcodec, default_ffmpeg_pix_fmt, default_ffmpeg_exe_path

    ffmpeg_exc_path = os.environ.get('ffmpeg_exe_path', default_ffmpeg_exe_path)
    vcodec = os.environ.get('ffmpeg_vcodec', default_ffmpeg_vcodec)
    pix_fmt = os.environ.get('ffmpeg_pix_fmt', default_ffmpeg_pix_fmt)

    if suffix is None:
        images_names = os.listdir(frames_dir)
        image_name = images_names[0]
        suffix = image_name.split('.')[-1]

    cmd = [
        ffmpeg_exc_path,
        '-y',
        '-r', str(fps),
        '-f', 'image2',
        '-pattern_type', 'glob',
        '-i', f'{frames_dir}/{filter}.{suffix}',
        '-vcodec', vcodec,
        '-pix_fmt', pix_fmt,
        out_path
    ]  # yapf: disable
    print(' '.join(cmd))
    subprocess.call(cmd)

    return out_path
