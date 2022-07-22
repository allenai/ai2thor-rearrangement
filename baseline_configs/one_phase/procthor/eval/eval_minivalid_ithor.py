from baseline_configs.one_phase.procthor.ithor.ithor_one_phase_rgb_fine_tune import (
    OnePhaseRGBClipResNet50FineTuneExperimentConfig as BaseConfig,
)

import copy
import platform
from typing import Optional, List, Sequence

import ai2thor.platform
import torch

from allenact.base_abstractions.sensor import ExpertActionSensor, Sensor
from allenact.utils.misc_utils import partition_sequence, md5_hash_str_as_int
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_sensors import (
    BinnedPointCloudMapTHORSensor,
    SemanticMapTHORSensor,
)
from allenact_plugins.ithor_plugin.ithor_util import get_open_x_displays


class EvalLoaderMixin:
    DATA_PREFIX = ""
    SAMPLER_STAGE = "ithor_mini_val"

    @staticmethod
    def get_scenes(stage: str) -> List[str]:
        """Returns a list of iTHOR scene names for each stage."""
        assert stage in {
            "train",
            "train_unseen",
            "val",
            "valid",
            "test",
            "all",
            "ithor_mini_val",
            "debug",
        }

        if stage == "debug":
            return ["FloorPlan1"]

        # [1-20] for train, [21-25] for val, [26-30] for test
        if stage in ["train", "train_unseen"]:
            scene_nums = range(1, 21)
        elif stage in ["val", "valid", "ithor_mini_val"]:
            scene_nums = range(21, 26)
        elif stage == "test":
            scene_nums = range(26, 31)
        elif stage == "all":
            scene_nums = range(1, 31)
        else:
            raise NotImplementedError

        kitchens = [f"FloorPlan{i}" for i in scene_nums]
        living_rooms = [f"FloorPlan{200 + i}" for i in scene_nums]
        bedrooms = [f"FloorPlan{300 + i}" for i in scene_nums]
        bathrooms = [f"FloorPlan{400 + i}" for i in scene_nums]
        return kitchens + living_rooms + bedrooms + bathrooms

    @classmethod
    def prepare_stagewise_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        sensors: Sequence[Sensor],
        allowed_rearrange_inds_subset: Optional[Sequence[int]] = None,
        allowed_scenes: Sequence[str] = None,
        devices: Optional[List[int]] = None,
    ):
        stage = self.SAMPLER_STAGE

        if allowed_scenes is not None:
            scenes = allowed_scenes
        elif stage == "combined":
            # Split scenes more evenly as the train scenes will have more episodes
            train_scenes = self.get_scenes("train")
            other_scenes = self.get_scenes("val") + self.get_scenes("test")
            assert len(train_scenes) == 2 * len(other_scenes)
            scenes = []
            while len(train_scenes) != 0:
                scenes.append(train_scenes.pop())
                scenes.append(train_scenes.pop())
                scenes.append(other_scenes.pop())
            assert len(train_scenes) == len(other_scenes)
        else:
            scenes = self.get_scenes(self.DATA_PREFIX + stage)

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

        kwargs = dict(
            stage=stage,
            allowed_scenes=allowed_scenes,
            scene_to_allowed_rearrange_inds=scene_to_allowed_rearrange_inds,
            seed=seed,
            x_display=x_display,
            thor_controller_kwargs=dict(gpu_device=gpu_device, platform=thor_platform,),
        )

        kwargs["sensors"] = list(sensors)

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


class EvalConfig(BaseConfig):
    EVAL_MIXIN = EvalLoaderMixin

    @classmethod
    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices=None,
        seeds=None,
        deterministic_cudnn: bool = False,
        task_spec_in_metrics: bool = False,
    ):
        return dict(
            force_cache_reset=True,
            epochs=1,
            task_spec_in_metrics=False,
            **self.EVAL_MIXIN.prepare_stagewise_task_sampler_args(
                process_ind=process_ind,
                total_processes=total_processes,
                sensors=copy.deepcopy(self.sensors()),
                devices=devices,
            ),
        )
