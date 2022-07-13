from baseline_configs.one_phase.procthor.one_phase_rgb_clip_dagger import (
    ProcThorOnePhaseRGBClipResNet50DaggerTrainMultiNodeConfig as BaseConfig,
)

import os
import copy
import platform
from typing import Optional, List, Sequence, Dict
import glob

import ai2thor.platform
import torch
import compress_json
import compress_pickle

from allenact.base_abstractions.sensor import ExpertActionSensor, Sensor, SensorSuite
from allenact.utils.misc_utils import partition_sequence, md5_hash_str_as_int
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_sensors import (
    BinnedPointCloudMapTHORSensor,
    SemanticMapTHORSensor,
)
from allenact_plugins.ithor_plugin.ithor_util import get_open_x_displays

from rearrange_constants import ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR
from rearrange.procthor_rearrange.tasks import RearrangeTaskSampler


def get_scenes(split_folder="split_mini_val"):
    base_dir = os.path.join("data", "2022procthor")

    scene_names_file = os.path.join(
        ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR,
        base_dir,
        split_folder,
        "scene_names.json.gz",
    )
    if os.path.exists(scene_names_file):
        print(f"Cached scenes file found at {scene_names_file}, using this file.")
        return compress_json.load(scene_names_file)

    firsts_lasts_fnames = []
    dataset_file_paths = glob.glob(os.path.join(base_dir, split_folder, "*.pkl.gz"))
    assert len(dataset_file_paths) > 0
    for fname in dataset_file_paths:
        vals = fname.replace(".pkl.gz", "").split("_")[-2:]
        firsts_lasts_fnames.append((int(vals[0]), int(vals[1]), fname))
    firsts_lasts_fnames = sorted(firsts_lasts_fnames)

    scenes = []
    for first, last, fname in firsts_lasts_fnames:
        get_logger().info(f"Loading data from {fname}")
        current_scenes = list(compress_pickle.load(path=fname).keys())
        scenes.extend(sorted(current_scenes, key=lambda x: int(x.split("_")[1])))

    scenes = sorted(scenes, key=lambda x: int(x.split("_")[1]))

    compress_json.dump(scenes, scene_names_file)
    return scenes


class EvalConfig(BaseConfig):
    def stagewise_task_sampler_args(
        self,
        stage: str,
        process_ind: int,
        total_processes: int,
        allowed_rearrange_inds_subset: Optional[Sequence[int]] = None,
        allowed_scenes: Sequence[str] = None,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ):
        if allowed_scenes is not None:
            scenes = allowed_scenes
        elif stage == "combined":
            # Split scenes more evenly as the train scenes will have more episodes
            train_scenes = get_scenes("train")
            other_scenes = get_scenes("val") + get_scenes("test")
            assert len(train_scenes) == 2 * len(other_scenes)
            scenes = []
            while len(train_scenes) != 0:
                scenes.append(train_scenes.pop())
                scenes.append(train_scenes.pop())
                scenes.append(other_scenes.pop())
            assert len(train_scenes) == len(other_scenes)
        else:
            scenes = get_scenes("split_" + stage)

        if total_processes > len(scenes):
            assert stage == "train" and total_processes % len(scenes) == 0
            scenes = scenes * (total_processes // len(scenes))

        allowed_scenes = list(
            sorted(partition_sequence(seq=scenes, parts=total_processes,)[process_ind])
        )

        scene_to_allowed_rearrange_inds = None
        if allowed_rearrange_inds_subset is not None:
            allowed_rearrange_inds_subset = tuple(allowed_rearrange_inds_subset)
            assert stage in ["valid", "train_unseen"]
            scene_to_allowed_rearrange_inds = {
                scene: allowed_rearrange_inds_subset for scene in allowed_scenes
            }
        seed = md5_hash_str_as_int(str(allowed_scenes))

        device = (
            devices[process_ind % len(devices)]
            if devices is not None and len(devices) > 0
            else torch.device("cpu")
        )
        x_display: Optional[str] = None
        gpu_device: Optional[int] = None
        thor_platform: Optional[ai2thor.platform.BaseLinuxPlatform] = None
        if platform.system() == "Linux":
            try:
                x_displays = get_open_x_displays(throw_error_if_empty=True)

                if devices is not None and len(
                    [d for d in devices if d != torch.device("cpu")]
                ) > len(x_displays):
                    get_logger().warning(
                        f"More GPU devices found than X-displays (devices: `{x_displays}`, x_displays: `{x_displays}`)."
                        f" This is not necessarily a bad thing but may mean that you're not using GPU memory as"
                        f" efficiently as possible. Consider following the instructions here:"
                        f" https://allenact.org/installation/installation-framework/#installation-of-ithor-ithor-plugin"
                        f" describing how to start an X-display on every GPU."
                    )
                x_display = x_displays[process_ind % len(x_displays)]
            except IOError:
                # Could not find an open `x_display`, use CloudRendering instead.
                assert all(
                    [d != torch.device("cpu") and d >= 0 for d in devices]
                ), "Cannot use CPU devices when there are no open x-displays as CloudRendering requires specifying a GPU."
                gpu_device = device
                thor_platform = ai2thor.platform.CloudRendering

        kwargs = {
            "stage": stage,
            "allowed_scenes": allowed_scenes,
            "scene_to_allowed_rearrange_inds": scene_to_allowed_rearrange_inds,
            "seed": seed,
            "x_display": x_display,
            "thor_controller_kwargs": {
                "gpu_device": gpu_device,
                "platform": thor_platform,
            },
        }

        sensors = kwargs.get("sensors", copy.deepcopy(self.sensors()))
        kwargs["sensors"] = sensors

        sem_sensor = next(
            (s for s in kwargs["sensors"] if isinstance(s, SemanticMapTHORSensor)), None
        )
        binned_pc_sensor = next(
            (
                s
                for s in kwargs["sensors"]
                if isinstance(s, BinnedPointCloudMapTHORSensor)
            ),
            None,
        )

        if sem_sensor is not None:
            sem_sensor.device = torch.device(device)

        if binned_pc_sensor is not None:
            binned_pc_sensor.device = torch.device(device)

        if stage != "train":
            # Don't include several sensors during validation/testing
            kwargs["sensors"] = [
                s
                for s in kwargs["sensors"]
                if not isinstance(
                    s,
                    (
                        ExpertActionSensor,
                        SemanticMapTHORSensor,
                        BinnedPointCloudMapTHORSensor,
                    ),
                )
            ]
        return kwargs

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices=None,
        seeds=None,
        deterministic_cudnn: bool = False,
        task_spec_in_metrics: bool = False,
    ):
        task_spec_in_metrics = False

        # Train_unseen
        # stage = "train_unseen"
        # allowed_rearrange_inds_subset = list(range(15))

        # Val
        stage = "mini_val"
        allowed_rearrange_inds_subset = None

        # Test
        # stage = "test"
        # allowed_rearrange_inds_subset = None

        # Combined (Will run inference on all datasets)
        # stage = "combined"
        # allowed_rearrange_inds_subset = None

        return dict(
            force_cache_reset=True,
            epochs=1,
            task_spec_in_metrics=task_spec_in_metrics,
            **self.stagewise_task_sampler_args(
                stage=stage,
                allowed_rearrange_inds_subset=allowed_rearrange_inds_subset,
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            ),
        )

    def make_sampler_fn(
        self,
        stage: str,
        force_cache_reset: bool,
        allowed_scenes: Optional[Sequence[str]],
        seed: int,
        epochs: int,
        scene_to_allowed_rearrange_inds: Optional[Dict[str, Sequence[int]]] = None,
        x_display: Optional[str] = None,
        sensors: Optional[Sequence[Sensor]] = None,
        thor_controller_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> RearrangeTaskSampler:
        """Return a RearrangeTaskSampler."""
        sensors = self.sensors() if sensors is None else sensors
        if "mp_ctx" in kwargs:
            del kwargs["mp_ctx"]
        assert not self.RANDOMIZE_START_ROTATION_DURING_TRAINING
        if not os.path.exists("data/2022procthor/mini_val_houses.pkl.gz"):
            raise ValueError("Please call `inv make-valid-houses-file")
        return RearrangeTaskSampler.from_fixed_dataset(
            run_walkthrough_phase=False,
            run_unshuffle_phase=True,
            stage=stage,
            allowed_scenes=allowed_scenes,
            scene_to_allowed_rearrange_inds=scene_to_allowed_rearrange_inds,
            rearrange_env_kwargs=dict(
                force_cache_reset=force_cache_reset,
                **self.REARRANGE_ENV_KWARGS,
                controller_kwargs={
                    "x_display": x_display,
                    **self.THOR_CONTROLLER_KWARGS,
                    **(
                        {} if thor_controller_kwargs is None else thor_controller_kwargs
                    ),
                    "renderDepthImage": False,
                },
                valid_houses_file="data/2022procthor/mini_val_houses.pkl.gz"
                if stage == "mini_val"
                else None,
            ),
            seed=seed,
            sensors=SensorSuite(sensors),
            max_steps=self.MAX_STEPS,
            discrete_actions=self.actions(),
            require_done_action=self.REQUIRE_DONE_ACTION,
            force_axis_aligned_start=self.FORCE_AXIS_ALIGNED_START,
            epochs=epochs,
            expert_exploration_enabled=self.EXPERT_EXPLORATION_ENABLED,
            **kwargs,
        )
