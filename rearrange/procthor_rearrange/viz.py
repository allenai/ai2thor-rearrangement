import os
import argparse
import glob
import math

import cv2
import torch
import numpy as np
import ffmpeg

from allenact.utils.experiment_utils import set_seed
from allenact.utils.inference import InferenceAgent
from allenact.algorithms.onpolicy_sync.runner import OnPolicyRunner

from rearrange.utils import save_frames_to_mp4

from baseline_configs.one_phase.procthor.one_phase_rgb_clip_dagger import (
    ProcThorOnePhaseRGBClipResNet50DaggerTrainMultiNodeConfig as TrainConfig,
)
from baseline_configs.one_phase.procthor.eval.eval_minivalid_procthor import (
    EvalConfig as ValidConfig,
)
from utils.table_visualizer import TableVisualizer


def topdown_setup(task, height=19.0, y_min=0.0, y_max=3.5):
    props = task.unshuffle_env.controller.step("GetMapViewCameraProperties").metadata[
        "actionReturn"
    ]

    autofov = 2 * np.degrees(
        np.arctan(props["orthographicSize"] / (height + y_min - y_max))
    )

    props["position"]["y"] = height + y_min
    props["fov"] = autofov

    task.unshuffle_env.controller.step(
        action="AddThirdPartyCamera",
        position=props["position"],
        rotation=props["rotation"],
        fieldOfView=props["fov"],
    )

    task.walkthrough_env.controller.step(
        action="AddThirdPartyCamera",
        position=props["position"],
        rotation=props["rotation"],
        fieldOfView=props["fov"],
    )

    return props


def extract_views(task):
    res = {}
    unshuffle_obs = task.unshuffle_env.observation
    res["unshuffle"] = {
        "rgb": unshuffle_obs[0],
        "depth": unshuffle_obs[1],
    }
    walkthrough_obs = task.walkthrough_env.observation
    res["walkthrough"] = {
        "rgb": walkthrough_obs[0],
        "depth": walkthrough_obs[1],
    }
    task.unshuffle_env.controller.step(action="ToggleMapView")
    res["topdown"] = task.unshuffle_env.controller.last_event.third_party_camera_frames[
        0
    ]
    task.unshuffle_env.controller.step(action="ToggleMapView")
    task.walkthrough_env.controller.step(action="ToggleMapView")
    res["goal"] = task.walkthrough_env.controller.last_event.third_party_camera_frames[
        0
    ]
    task.walkthrough_env.controller.step(action="ToggleMapView")

    return res


def make_frame(task, action_name=""):
    views = extract_views(task)

    frame = np.concatenate(
        (
            np.concatenate(
                (views["unshuffle"]["rgb"], views["walkthrough"]["rgb"]), axis=1
            ),
            np.concatenate((views["topdown"], views["goal"]), axis=1),
        ),
        axis=0,
    )

    for color, thickness in zip([(0, 0, 0), (255, 255, 255)], [2, 1]):
        frame = cv2.putText(
            frame,
            action_name,
            (10, 20),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            color,
            thickness=thickness,
        )

    return frame


def run_task(
    agent, task, stage, ind, folder, viz=None, checkpoint=None, episode_it=None
):
    print(f"Task {stage} {ind} starting")
    observations = task.get_observations()

    topdown_setup(task)

    frames = []
    nframes = 0
    while not task.is_done():
        action = agent.act(observations=observations)

        action_name = task.action_names()[action]
        msg = f"{nframes} {action_name}"
        frame = make_frame(task, msg)
        print(msg)

        step_result = task.step(action)
        observations = step_result.observation
        if step_result.info["action_success"]:
            frames.append(frame)
        else:
            frames.append(
                cv2.rectangle(
                    frame,
                    (0, 0),
                    tuple(s - 1 for s in frame.shape[:2]),
                    (255, 0, 0),
                    5,
                )
            )

        nframes += 1

    # Append final scene state
    frames.append(make_frame(task, ""))

    if viz is not None:
        assert checkpoint is not None
        assert episode_it is not None

        metrics = task.metrics()
        metrics["task_info"].pop("unshuffle_actions", None)
        metrics["task_info"].pop("unshuffle_action_successes", None)
        old_keys = list(metrics.keys())
        for key in old_keys:
            if key.startswith("unshuffle/"):
                metrics[key.replace("unshuffle/", "")] = metrics.pop(key, None)
        return (
            viz.save_video_get_path(frames, checkpoint_steps(checkpoint), episode_it),
            metrics,
        )
    else:
        # Last frame seems to be missing after save_frames_to_mp4 - try duplicating it
        frames.append(frames[-1])

        file_name = os.path.join(folder, f"rearrange_{ind}.mp4")
        print(f"Saving trajectory to {file_name}")
        save_frames_to_mp4(frames=frames, file_name=file_name)


def next_task(sampler, current_ind=0, next_ind=0):
    for _ in range(next_ind - current_ind - 1):
        try:
            next(sampler.task_spec_iterator)
        except:
            return None, None
    return sampler.next_task(), next_ind


def checkpoint_steps(checkpoint_file):
    return int(os.path.basename(checkpoint_file).split("__")[-1][:-3].split("_")[-1])


def setup_processor(base_output_folder, stage, checkpoint_file):
    set_seed(12345)

    folder = os.path.join(
        base_output_folder,
        stage,
        os.path.basename(checkpoint_file).split("__")[-1][:-3],
    )
    os.makedirs(folder, exist_ok=True)

    if stage == "train":
        config_class = TrainConfig
    elif stage == "valid":
        config_class = ValidConfig
    else:
        raise NotImplementedError

    config = config_class()
    agent = InferenceAgent.from_experiment_config(
        exp_config=config,
        device=torch.device("cuda:0") if torch.cuda.device_count() > 0 else "cpu",
        checkpoint_path=checkpoint_file,
    )

    sampler = config.make_sampler_fn(
        **config.test_task_sampler_args(process_ind=0, total_processes=1, seeds=[12345])
    )
    sampler.reset()

    return folder, agent, sampler


def run(
    checkpoint_file,
    stage="train",
    base_output_folder="rearrange_viz",
    first=0,
    last=1000,
    jump=50,
    viz=None,
    inds=None,
):
    folder, agent, sampler = setup_processor(base_output_folder, stage, checkpoint_file)

    old_ind = 0
    paths = []
    metrics = []

    if inds is None:
        iterator = range(first, last, jump)
    else:
        iterator = inds

    for episode_it, ind in enumerate(iterator):
        agent.reset()

        task, old_ind = next_task(sampler, old_ind, ind)
        if task is None:
            break

        path, metric = run_task(
            agent,
            task,
            stage,
            ind,
            folder,
            viz=viz,
            episode_it=episode_it,
            checkpoint=checkpoint_file,
        )
        paths.append(path)
        metrics.append(f"{metric}")

        agent.rollout_storage.after_updates()

    if viz is not None:
        viz.add_row(f"{checkpoint_steps(checkpoint_file):,}", paths, metrics)


def run_on_1_3_5(ckpt_file, output_folder):
    # 49 for 1 obj (no open), 25 for 3 objs (no open), 0 for 5 objs (1 open)
    for first in [49, 25, 0]:
        run(
            checkpoint_file=ckpt_file,
            stage="valid",
            base_output_folder=output_folder,
            first=first,
            last=1000,
            jump=50,
        )


def run_with_table_viz(
    output_folder, checkpoint_files, use_episodes=20, stage_episodes=1000
):
    print(
        f"Visualizing episodes for steps {[OnPolicyRunner.step_from_checkpoint(p) for p in checkpoint_files]}"
    )
    viz = HtmlViz(output_folder, use_episodes)
    for ckpt_file in checkpoint_files:
        print(f"{ckpt_file}")
        run(
            checkpoint_file=ckpt_file,
            stage="valid",
            base_output_folder=output_folder,
            viz=viz,
            inds=np.round(np.linspace(0, stage_episodes - 1, num=use_episodes))
            .astype(np.int64)
            .tolist(),
        )


# From https://github.com/kylemcdonald/python-utils/blob/master/ffmpeg.py
class VideoWriter:
    def __init__(
        self,
        fn,
        vcodec="libx264",
        fps=3,
        in_pix_fmt="rgb24",
        out_pix_fmt="yuv420p",
        input_args=None,
        output_args=None,
    ):
        self.fn = fn
        self.process = None
        self.input_args = {} if input_args is None else input_args
        self.output_args = (
            dict(
                crf=20, preset="veryslow", movflags="faststart"
            )  # try crf=17 for near-lossless quality
            if output_args is None
            else output_args
        )
        self.input_args["framerate"] = fps
        self.input_args["pix_fmt"] = in_pix_fmt
        self.output_args["pix_fmt"] = out_pix_fmt
        self.output_args["vcodec"] = vcodec

    def add(self, frame):
        if self.process is None:
            h, w = frame.shape[:2]
            self.process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    s="{}x{}".format(w, h),
                    **self.input_args,
                )
                .output(self.fn, **self.output_args)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
        self.process.stdin.write(frame.astype(np.uint8).tobytes())

    def close(self):
        if self.process is None:
            return
        self.process.stdin.close()
        self.process.wait()


def vidwrite(fn, images, **kwargs):
    writer = VideoWriter(fn, **kwargs)
    for image in images:
        writer.add(image)
    writer.close()


class HtmlViz:
    def __init__(self, path, episodes_per_checkpoint=20):
        self.path = path
        self.episodes_per_checkpoint = episodes_per_checkpoint

        if not os.path.exists(self.path):
            os.mkdir(self.path)

        self.video_folder = os.path.join(self.path, "videos")
        os.makedirs(self.video_folder, exist_ok=True)

        out_file_name = os.path.join(self.path, "viz.html")

        table_configs = []
        table_configs.append(
            {"id": "checkpoint", "display_name": "Checkpoint", "type": "text"}
        )
        for it in range(self.episodes_per_checkpoint):
            table_configs.append(
                {
                    "id": f"unshuffle_trajectory_{it}",
                    "display_name": f"Unshuffle Trajectory {it}",
                    "type": "video",
                    "height": 200,
                }
            )
            table_configs.append(
                {
                    "id": f"unshuffle_metrics_{it}",
                    "display_name": f"Unshuffle Metrics {it}",
                    "type": "text",
                }
            )

        self.table_viz = TableVisualizer(
            table_configs=table_configs, out_file_name=out_file_name
        )

    def save_video_get_path(self, frames, checkpoint_steps, episode_it):
        basename = f"unshuffle__checkpoint{checkpoint_steps:_}__episode{episode_it}.mp4"
        vidwrite(fn=os.path.join(self.video_folder, basename), images=frames)
        return os.path.join("videos", basename)

    def add_row(self, checkpoint, video_paths, metrics, reverse_order=True):
        row_viz = [checkpoint]

        if len(video_paths) > self.episodes_per_checkpoint:
            print(
                f"WARNING: {len(video_paths)} inputs for {self.episodes_per_checkpoint} outputs."
                f" Discarding last inputs."
            )

        if reverse_order:
            metric, video_paths = (
                metrics[::-1][-self.episodes_per_checkpoint :],
                video_paths[::-1][-self.episodes_per_checkpoint :],
            )

        for it in range(self.episodes_per_checkpoint):
            if len(video_paths) > it:
                row_viz.append(video_paths[it])
                row_viz.append(metrics[it])
            else:
                row_viz.append("None")
                row_viz.append("None")

        self.table_viz.add_row(row_viz)

        # Update html
        self.table_viz.render()


def arg_parse():
    """Parses arguments"""

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description="rearrange-viz",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        required=True,
        default=None,
        type=str,
        help="checkpoint file name. This argument can be used very flexibly as:"
        "\n(1) the path to a particular individual checkpoint file,"
        "\n(2) the path to a directory of checkpoint files all of which you'd like to be visualized"
        " (checkpoints are expected to have a `.pt` file extension),"
        '\n(3) a "glob" pattern (https://tldp.org/LDP/abs/html/globbingref.html) that will be expanded'
        " using python's `glob.glob` function and should return a collection of checkpoint files."
        "\nIf you'd like to only visualize a subset of the checkpoints specified by the above directory/glob"
        " (e.g. every checkpoint saved after 5mil steps) you'll likely want to use the `--approx_ckpt_step_interval`"
        " flag.",
    )

    parser.add_argument(
        "--approx_ckpt_step_interval",
        required=False,
        default=None,
        type=float,
        help="if visualizin a collection of checkpoints (see the `--checkpoint` flag) this argument can be"
        " used to skip checkpoints. In particular, if this value is specified and equals `n` then we will"
        " only visualize checkpoints whose step count is closest to each of `0*n`, `1*n`, `2*n`, `3*n`, ... "
        " n * ceil(max training steps in ckpts / n).",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=str,
        default="rearrange_visualization",
        help="visualization output folder",
    )

    return parser.parse_args()


def get_checkpoint_files(
    checkpoint_path_dir_or_pattern: str, approx_ckpt_step_interval=None,
):
    if os.path.isdir(checkpoint_path_dir_or_pattern):
        # The fragment is a path to a directory, lets use this directory
        # as the base dir to search for checkpoints
        checkpoint_path_dir_or_pattern = os.path.join(
            checkpoint_path_dir_or_pattern, "*.pt"
        )

    ckpt_paths = glob.glob(checkpoint_path_dir_or_pattern, recursive=True)

    if len(ckpt_paths) == 0:
        raise FileNotFoundError(
            f"Could not find any checkpoints at {os.path.abspath(checkpoint_path_dir_or_pattern)}, is it possible"
            f" the path has been mispecified?"
        )

    step_count_ckpt_pairs = [
        (OnPolicyRunner.step_from_checkpoint(p), p) for p in ckpt_paths
    ]
    step_count_ckpt_pairs.sort()
    ckpts_paths = [p for _, p in step_count_ckpt_pairs]
    step_counts = np.array([sc for sc, _ in step_count_ckpt_pairs])

    if approx_ckpt_step_interval is not None:
        assert approx_ckpt_step_interval > 0, "`approx_ckpt_step_interval` must be >0"
        inds_to_eval = set()
        for i in range(
            math.ceil(step_count_ckpt_pairs[-1][0] / approx_ckpt_step_interval) + 1
        ):
            inds_to_eval.add(
                int(np.argmin(np.abs(step_counts - i * approx_ckpt_step_interval)))
            )

        ckpts_paths = [ckpts_paths[ind] for ind in sorted(list(inds_to_eval))]
    return ckpts_paths


if __name__ == "__main__":
    args = arg_parse()

    run_with_table_viz(
        output_folder=args.output_dir,
        checkpoint_files=get_checkpoint_files(
            args.checkpoint, args.approx_ckpt_step_interval
        ),
    )

    print("DONE")
