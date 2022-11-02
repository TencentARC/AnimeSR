# The codes below partially refer to the PySceneDetect. According
# to its BSD 3-Clause License, we keep the following.
#
#          PySceneDetect: Python-Based Video Scene Detector
#   ---------------------------------------------------------------
#     [  Site: http://www.bcastell.com/projects/PySceneDetect/   ]
#     [  Github: https://github.com/Breakthrough/PySceneDetect/  ]
#     [  Documentation: http://pyscenedetect.readthedocs.org/    ]
#
# Copyright (C) 2014-2020 Brandon Castellano <http://www.bcastell.com>.
#
# PySceneDetect is licensed under the BSD 3-Clause License; see the included
# LICENSE file, or visit one of the following pages for details:
#  - https://github.com/Breakthrough/PySceneDetect/
#  - http://www.bcastell.com/projects/PySceneDetect/
#
# This software uses Numpy, OpenCV, click, tqdm, simpletable, and pytest.
# See the included LICENSE files or one of the above URLs for more information.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import cv2
import glob
import numpy as np
import os
from tqdm import tqdm

DEFAULT_DOWNSCALE_FACTORS = {
    3200: 12,  # ~4k
    2100: 8,  # ~2k
    1700: 6,  # ~1080p
    1200: 5,
    900: 4,  # ~720p
    600: 3,
    400: 1  # ~480p
}


def compute_downscale_factor(frame_width):
    """Compute Downscale Factor: Returns the optimal default downscale factor
    based on a video's resolution (specifically, the width parameter).

    Returns:
        int: The defalt downscale factor to use with a video of frame_height x
            frame_width.
    """
    for width in sorted(DEFAULT_DOWNSCALE_FACTORS, reverse=True):
        if frame_width >= width:
            return DEFAULT_DOWNSCALE_FACTORS[width]
    return 1


class ShotDetector(object):
    """Detects fast cuts using changes in colour and intensity between frames.

    Detect shot boundary using HSV and LUV information.
    """

    def __init__(self, threshold=30.0, min_shot_len=15):
        super(ShotDetector, self).__init__()
        self.hsv_threshold = threshold
        self.delta_hsv_gap_threshold = 10
        self.luv_threshold = 40
        self.hsv_weight = 5
        # minimum length (frames length) of any given shot
        self.min_shot_len = min_shot_len
        self.last_frame = None
        self.last_shot_cut = None
        self.last_hsv = None
        self._metric_keys = [
            'hsv_content_val', 'delta_hsv_hue', 'delta_hsv_sat', 'delta_hsv_lum', 'luv_content_val', 'delta_luv_hue',
            'delta_luv_sat', 'delta_luv_lum'
        ]
        self.cli_name = 'detect-content'
        self.last_luv = None
        self.cut_list = []

    def add_cut(self, cut):
        num_cuts = len(self.cut_list)
        if num_cuts == 0:
            self.cut_list.append([0, cut - 1])
        else:
            self.cut_list.append([self.cut_list[num_cuts - 1][1] + 1, cut - 1])

    def process_frame(self, frame_num, frame_img):
        """Similar to ThresholdDetector, but using the HSV colour space
        DIFFERENCE instead of single-frame RGB/grayscale intensity (thus cannot
        detect slow fades with this method).

        Args:
            frame_num (int): Frame number of frame that is being passed.

            frame_img (Optional[int]): Decoded frame image (np.ndarray) to
                perform shot detection on. Can be None *only* if the
                self.is_processing_required() method
                (inhereted from the base shotDetector class) returns True.

        Returns:
            List[int]: List of frames where shot cuts have been detected.
            There may be 0 or more frames in the list, and not necessarily
            the same as frame_num.
        """
        cut_list = []

        # if self.last_frame is not None:
        #     # Change in average of HSV (hsv), (h)ue only,
        #     # (s)aturation only, (l)uminance only.
        delta_hsv_avg, delta_hsv_h, delta_hsv_s, delta_hsv_v = 0.0, 0.0, 0.0, 0.0
        delta_luv_avg, delta_luv_h, delta_luv_s, delta_luv_v = 0.0, 0.0, 0.0, 0.0
        if frame_num == 0:
            self.last_frame = frame_img.copy()
            return cut_list

        else:
            num_pixels = frame_img.shape[0] * frame_img.shape[1]
            curr_luv = cv2.split(cv2.cvtColor(frame_img, cv2.COLOR_BGR2Luv))
            curr_hsv = cv2.split(cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV))
            last_hsv = self.last_hsv
            last_luv = self.last_luv
            if not last_hsv:
                last_hsv = cv2.split(cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2HSV))
                last_luv = cv2.split(cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2Luv))

            delta_hsv = [0, 0, 0, 0]
            for i in range(3):
                num_pixels = curr_hsv[i].shape[0] * curr_hsv[i].shape[1]
                curr_hsv[i] = curr_hsv[i].astype(np.int32)
                last_hsv[i] = last_hsv[i].astype(np.int32)
                delta_hsv[i] = np.sum(np.abs(curr_hsv[i] - last_hsv[i])) / float(num_pixels)
            delta_hsv[3] = sum(delta_hsv[0:3]) / 3.0
            delta_hsv_h, delta_hsv_s, delta_hsv_v, delta_hsv_avg = \
                delta_hsv

            delta_luv = [0, 0, 0, 0]
            for i in range(3):
                num_pixels = curr_luv[i].shape[0] * curr_luv[i].shape[1]
                curr_luv[i] = curr_luv[i].astype(np.int32)
                last_luv[i] = last_luv[i].astype(np.int32)
                delta_luv[i] = np.sum(np.abs(curr_luv[i] - last_luv[i])) / float(num_pixels)
            delta_luv[3] = sum(delta_luv[0:3]) / 3.0
            delta_luv_h, delta_luv_s, delta_luv_v, delta_luv_avg = \
                delta_luv

            self.last_hsv = curr_hsv
            self.last_luv = curr_luv
        if delta_hsv_avg >= self.hsv_threshold and delta_hsv_avg - self.hsv_threshold >= self.delta_hsv_gap_threshold:
            if self.last_shot_cut is None or ((frame_num - self.last_shot_cut) >= self.min_shot_len):
                cut_list.append(frame_num)
                self.last_shot_cut = frame_num
        elif delta_hsv_avg >= self.hsv_threshold and \
                delta_hsv_avg - self.hsv_threshold < \
                self.delta_hsv_gap_threshold and \
                delta_luv_avg + self.hsv_weight * \
                (delta_hsv_avg - self.hsv_threshold) > self.luv_threshold:
            if self.last_shot_cut is None or ((frame_num - self.last_shot_cut) >= self.min_shot_len):
                cut_list.append(frame_num)
                self.last_shot_cut = frame_num

        self.last_frame = frame_img.copy()
        return cut_list

    def detect_shots(self, frame_source, frame_skip=0, show_progress=True, keep_resolution=False):
        """Perform shot detection on the given frame_source using the added
            shotDetectors.

            Blocks until all frames in the frame_source have been processed.
            Results can be obtained by calling either the get_shot_list()
            or get_cut_list() methods.
            Arguments:
                frame_source (shotdetect.video_manager.VideoManager or
                    cv2.VideoCapture):
                    A source of frames to process (using frame_source.read() as in
                    VideoCapture).
                    VideoManager is preferred as it allows concatenation of
                    multiple videos as well as seeking, by defining start time
                    and end time/duration.
                end_time (int or FrameTimecode): Maximum number of frames to detect
                    (set to None to detect all available frames). Only needed for
                    OpenCV
                    VideoCapture objects; for VideoManager objects, use
                    set_duration() instead.
                frame_skip (int): Not recommended except for extremely high
                    framerate videos.
                    Number of frames to skip (i.e. process every 1 in N+1 frames,
                    where N is frame_skip, processing only 1/N+1 percent of the
                    video,
                    speeding up the detection time at the expense of accuracy).
                    `frame_skip` **must** be 0 (the default) when using a
                    StatsManager.
                show_progress (bool): If True, and the ``tqdm`` module is
                    available, displays
                    a progress bar with the progress, framerate, and expected
                    time to
                    complete processing the video frame source.

            Raises:
                ValueError: `frame_skip` **must** be 0 (the default)
                    if the shotManager
                    was constructed with a StatsManager object.
            """

        if frame_skip > 0 and self._stats_manager is not None:
            raise ValueError('frame_skip must be 0 when using a StatsManager.')

        curr_frame = 0
        frame_paths = sorted(glob.glob(os.path.join(frame_source, '*')))
        total_frames = len(frame_paths)
        end_frame = total_frames

        progress_bar = None
        if tqdm and show_progress:
            progress_bar = tqdm(total=total_frames, unit='frames')

        try:
            while True:
                if end_frame is not None and curr_frame >= end_frame:
                    break

                frame_im = cv2.imread(frame_paths[curr_frame])
                if not keep_resolution:
                    if curr_frame == 0:
                        downscale_factor = compute_downscale_factor(frame_im.shape[1])
                    frame_im = frame_im[::downscale_factor, ::downscale_factor, :]

                cut = self.process_frame(curr_frame, frame_im)

                if len(cut) != 0:
                    self.add_cut(cut[0])

                curr_frame += 1
                if progress_bar:
                    progress_bar.update(1)

        finally:
            if progress_bar:
                progress_bar.close()

        return self.cut_list
