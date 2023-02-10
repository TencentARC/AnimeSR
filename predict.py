import os
import shutil
import tempfile
from subprocess import call
from zipfile import ZipFile
from typing import Optional
import mimetypes
import torch

from cog import BasePredictor, Input, Path, BaseModel


call("python setup.py develop", shell=True)


class ModelOutput(BaseModel):
    video: Path
    sr_frames: Optional[Path]


class Predictor(BasePredictor):
    @torch.inference_mode()
    def predict(
        self,
        video: Path = Input(
            description="Input video file",
            default=None,
        ),
        frames: Path = Input(
            description="Zip file of frames of a video. Ignored when video is provided.",
            default=None,
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""
        assert frames or video, "Please provide frames of video input."

        out_path = "cog_temp"
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path, exist_ok=True)

        if video:
            print("processing video...")
            cmd = (
                "python scripts/inference_animesr_video.py -i "
                + str(video)
                + " -o "
                + out_path
                + " -n AnimeSR_v2 -s 4 --expname animesr_v2 --num_process_per_gpu 1"
            )
            call(cmd, shell=True)

        else:
            print("processing frames...")
            unzip_frames = "cog_frames_temp"
            if os.path.exists(unzip_frames):
                shutil.rmtree(unzip_frames)
            os.makedirs(unzip_frames)

            with ZipFile(str(frames), "r") as zip_ref:
                for zip_info in zip_ref.infolist():
                    if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                        "__MACOSX"
                    ):
                        continue
                    mt = mimetypes.guess_type(zip_info.filename)
                    if mt and mt[0] and mt[0].startswith("image/"):
                        zip_info.filename = os.path.basename(zip_info.filename)
                        zip_ref.extract(zip_info, unzip_frames)

            cmd = (
                "python scripts/inference_animesr_frames.py -i "
                + unzip_frames
                + " -o "
                + out_path
                + " -n AnimeSR_v2 --expname animesr_v2 --save_video_too --fps 20"
            )
            call(cmd, shell=True)

            frames_output = Path(tempfile.mkdtemp()) / "out.zip"
            frames_out_dir = os.listdir(f"{out_path}/animesr_v2/frames")
            assert len(frames_out_dir) == 1
            frames_path = os.path.join(
                f"{out_path}/animesr_v2/frames", frames_out_dir[0]
            )
            # by defult, sr_frames will be saved in cog_temp/animesr_v2/frames
            sr_frames_files = os.listdir(frames_path)

            with ZipFile(str(frames_output), "w") as zip:
                for img in sr_frames_files:
                    zip.write(os.path.join(frames_path, img))

        # by defult, video will be saved in cog_temp/animesr_v2/videos
        video_out_dir = os.listdir(f"{out_path}/animesr_v2/videos")
        assert len(video_out_dir) == 1
        if video_out_dir[0].endswith(".mp4"):
            source = os.path.join(f"{out_path}/animesr_v2/videos", video_out_dir[0])
        else:
            video_output = os.listdir(
                f"{out_path}/animesr_v2/videos/{video_out_dir[0]}"
            )[0]
            source = os.path.join(
                f"{out_path}/animesr_v2/videos", video_out_dir[0], video_output
            )
        video_path = Path(tempfile.mkdtemp()) / "out.mp4"
        shutil.copy(source, str(video_path))

        if video:
            return ModelOutput(video=video_path)
        return ModelOutput(sr_frames=frames_output, video=video_path)
