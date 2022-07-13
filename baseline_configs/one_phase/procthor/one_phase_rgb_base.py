import abc
import copy
import glob
import os
import platform
from abc import abstractmethod
from typing import Optional, List, Sequence, Dict, Any

import ai2thor.platform
import compress_pickle
import compress_json
import gym.spaces
import stringcase
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.base_abstractions.experiment_config import (
    ExperimentConfig,
    MachineParams,
    split_processes_onto_devices,
)
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.base_abstractions.sensor import SensorSuite, Sensor, ExpertActionSensor
from allenact.utils.experiment_utils import TrainingPipeline, LinearDecay, Builder
from allenact.utils.misc_utils import partition_sequence, md5_hash_str_as_int
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_util import get_open_x_displays
from rearrange.baseline_models import (
    RearrangeActorCriticSimpleConvRNN,
    ResNetRearrangeActorCriticRNN,
)
from rearrange.constants import OBJECT_TYPES_WITH_PROPERTIES
from rearrange.sensors import (
    RGBRearrangeSensor,
    UnshuffledRGBRearrangeSensor,
)
from rearrange.environment import RearrangeMode

from rearrange_constants import ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR
from rearrange.procthor_rearrange.tasks import RearrangeTaskSampler


def get_scenes():
    base_dir = os.path.join("data", "2022procthor")

    scene_names_file = os.path.join(
        ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR,
        base_dir,
        "split_train",
        "scene_names.json.gz",
    )
    if os.path.exists(scene_names_file):
        print(f"Cached scenes file found at {scene_names_file}, using this file.")
        return compress_json.load(scene_names_file)

    firsts_lasts_fnames = []
    dataset_file_paths = glob.glob(os.path.join(base_dir, "split_train", "*.pkl.gz"))
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


class OnePhaseRGBBaseExperimentConfig(ExperimentConfig, abc.ABC):
    PROCTHOR_SCENES = None

    EXPERT_EXPLORATION_ENABLED = False

    # Task parameters
    MAX_STEPS = {"walkthrough": 250, "unshuffle": 500}
    REQUIRE_DONE_ACTION = True
    FORCE_AXIS_ALIGNED_START = True
    RANDOMIZE_START_ROTATION_DURING_TRAINING = False

    # Environment parameters
    REARRANGE_ENV_KWARGS = dict(mode=RearrangeMode.SNAP,)
    SCREEN_SIZE = 224
    THOR_CONTROLLER_KWARGS = {
        "rotateStepDegrees": 90,
        "snapToGrid": True,
        "quality": "Very Low",
        "width": SCREEN_SIZE,
        "height": SCREEN_SIZE,
        "fastActionEmit": True,
        "scene": "Procedural",
    }
    INCLUDE_OTHER_MOVE_ACTIONS = False

    # Training parameters
    TRAINING_STEPS = int(1e9)
    SAVE_INTERVAL = int(1e6)
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")

    # Sensor info
    EGOCENTRIC_RGB_UUID = "rgb"
    UNSHUFFLED_RGB_UUID = "unshuffled_rgb"
    EGOCENTRIC_RGB_RESNET_UUID = "rgb_resnet"
    UNSHUFFLED_RGB_RESNET_UUID = "unshuffled_rgb_resnet"

    # Actions
    PICKUP_ACTIONS = list(
        sorted(
            [
                f"pickup_{stringcase.snakecase(object_type)}"
                for object_type, properties in OBJECT_TYPES_WITH_PROPERTIES.items()
                if properties["pickupable"]
            ]
        )
    )
    OPEN_ACTIONS = list(
        sorted(
            [
                f"open_by_type_{stringcase.snakecase(object_type)}"
                for object_type, properties in OBJECT_TYPES_WITH_PROPERTIES.items()
                if properties["openable"] and not properties["pickupable"]
            ]
        )
    )

    GRU_LAYERS = 1

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

    def actions(self):
        other_move_actions = (
            tuple()
            if not self.INCLUDE_OTHER_MOVE_ACTIONS
            else ("move_left", "move_right", "move_back",)
        )
        return (
            ("done", "move_ahead",)
            + other_move_actions
            + (
                "rotate_right",
                "rotate_left",
                "stand",
                "crouch",
                "look_up",
                "look_down",
                "drop_held_object_with_snap",
                *self.OPEN_ACTIONS,
                *self.PICKUP_ACTIONS,
            )
        )

    def sensors(self) -> Sequence[Sensor]:
        from allenact_plugins.clip_plugin.clip_preprocessors import (
            ClipResNetPreprocessor,
        )

        mean = ClipResNetPreprocessor.CLIP_RGB_MEANS
        stdev = ClipResNetPreprocessor.CLIP_RGB_STDS

        return [
            RGBRearrangeSensor(
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                use_resnet_normalization=True,
                uuid=self.EGOCENTRIC_RGB_UUID,
                mean=mean,
                stdev=stdev,
            ),
            UnshuffledRGBRearrangeSensor(
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                use_resnet_normalization=True,
                uuid=self.UNSHUFFLED_RGB_UUID,
                mean=mean,
                stdev=stdev,
            ),
        ]

    def resnet_preprocessor_graph(self, mode: str) -> SensorPreprocessorGraph:
        def create_resnet_builder(in_uuid: str, out_uuid: str):
            cnn_type, pretraining_type = self.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING
            if pretraining_type == "clip":
                from allenact_plugins.clip_plugin.clip_preprocessors import (
                    ClipResNetPreprocessor,
                )
                import clip

                # Let's make sure we download the clip model now
                # so we don't download it on every spawned process
                clip.load(cnn_type, "cpu")

                return ClipResNetPreprocessor(
                    rgb_input_uuid=in_uuid,
                    clip_model_type=cnn_type,
                    pool=False,
                    output_uuid=out_uuid,
                )
            else:
                raise NotImplementedError

        img_uuids = [self.EGOCENTRIC_RGB_UUID, self.UNSHUFFLED_RGB_UUID]
        return SensorPreprocessorGraph(
            source_observation_spaces=SensorSuite(
                [
                    sensor
                    for sensor in self.sensors()
                    if (mode == "train" or not isinstance(sensor, ExpertActionSensor))
                ]
            ).observation_spaces,
            preprocessors=[
                create_resnet_builder(sid, f"{sid}_resnet") for sid in img_uuids
            ],
        )

    def get_lr_scheduler_builder(self, use_lr_decay: bool):
        return (
            None
            if not use_lr_decay
            else Builder(
                LambdaLR,
                {
                    "lr_lambda": LinearDecay(
                        steps=self.TRAINING_STEPS // 3, startp=1.0, endp=1.0 / 3
                    )
                },
            )
        )

    def stagewise_task_sampler_args(
        self,
        stage: str,
        process_ind: int,
        total_processes: int,
        allowed_rearrange_inds_subset: Optional[Sequence[int]] = None,
        allowed_scenes: Sequence[str] = None,
        devices: Optional[Sequence[int]] = None,
        seeds: Optional[Sequence[int]] = None,
        deterministic_cudnn: bool = False,
    ):
        if allowed_scenes is not None:
            scenes = allowed_scenes
        elif stage == "combined":
            raise NotImplementedError
            # # Split scenes more evenly as the train scenes will have more episodes
            # train_scenes = datagen_utils.get_scenes("train")
            # other_scenes = datagen_utils.get_scenes("val") + datagen_utils.get_scenes(
            #     "test"
            # )
            # assert len(train_scenes) == 2 * len(other_scenes)
            # scenes = []
            # while len(train_scenes) != 0:
            #     scenes.append(train_scenes.pop())
            #     scenes.append(train_scenes.pop())
            #     scenes.append(other_scenes.pop())
            # assert len(train_scenes) == len(other_scenes)
        else:
            if self.PROCTHOR_SCENES is None:
                self.PROCTHOR_SCENES = get_scenes()
            scenes = self.PROCTHOR_SCENES

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
            except:  # IOError:
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

        # sem_sensor = next(
        #     (s for s in kwargs["sensors"] if isinstance(s, SemanticMapTHORSensor)), None
        # )
        # binned_pc_sensor = next(
        #     (
        #         s
        #         for s in kwargs["sensors"]
        #         if isinstance(s, BinnedPointCloudMapTHORSensor)
        #     ),
        #     None,
        # )
        #
        # if sem_sensor is not None:
        #     sem_sensor.device = torch.device(device)
        #
        # if binned_pc_sensor is not None:
        #     binned_pc_sensor.device = torch.device(device)

        if stage != "train":
            raise NotImplementedError
            # # Don't include several sensors during validation/testing
            # kwargs["sensors"] = [
            #     s
            #     for s in kwargs["sensors"]
            #     if not isinstance(
            #         s,
            #         (
            #             ExpertActionSensor,
            #             SemanticMapTHORSensor,
            #             BinnedPointCloudMapTHORSensor,
            #         ),
            #     )
            # ]

        return kwargs

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[Sequence[int]] = None,
        seeds: Optional[Sequence[int]] = None,
        deterministic_cudnn: bool = False,
    ):
        return dict(
            force_cache_reset=False,
            epochs=float("inf"),
            **self.stagewise_task_sampler_args(
                stage="train",
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            ),
        )

    def machine_params(self, mode="train", **kwargs):
        """Return the number of processes and gpu_ids to use with training."""
        num_gpus = torch.cuda.device_count()
        has_gpu = num_gpus != 0

        sampler_devices = None
        if mode == "train":
            nprocesses = self.num_train_processes() if torch.cuda.is_available() else 1
            devices = (
                list(range(min(nprocesses, num_gpus)))
                if has_gpu
                else [torch.device("cpu")]
            )
        elif mode == "valid":
            devices = [num_gpus - 1] if has_gpu else [torch.device("cpu")]
            nprocesses = 0 if has_gpu else 0
        else:
            nprocesses = 20 if has_gpu else 1
            devices = (
                list(range(min(nprocesses, num_gpus)))
                if has_gpu
                else [torch.device("cpu")]
            )

        nprocesses = split_processes_onto_devices(
            nprocesses=nprocesses, ndevices=len(devices)
        )

        return MachineParams(
            nprocesses=nprocesses,
            devices=devices,
            sampler_devices=sampler_devices,
            sensor_preprocessor_graph=self.resnet_preprocessor_graph(mode=mode)
            if self.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is not None
            else None,
        )

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ):
        return dict(
            force_cache_reset=True,
            epochs=1,
            **self.stagewise_task_sampler_args(
                stage="valid",
                allowed_rearrange_inds_subset=tuple(range(10)),
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            ),
        )

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
        task_spec_in_metrics: bool = False,
    ):
        task_spec_in_metrics = False

        # Train_unseen
        # stage = "train_unseen"
        # allowed_rearrange_inds_subset = list(range(15))

        # Val
        # stage = "val"
        # allowed_rearrange_inds_subset = None

        # Test
        # stage = "test"
        # allowed_rearrange_inds_subset = None

        # Combined (Will run inference on all datasets)
        stage = "combined"
        allowed_rearrange_inds_subset = None

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

    @abstractmethod
    def _training_pipeline_info(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def num_train_processes(self) -> int:
        raise NotImplementedError

    def training_pipeline(self, **kwargs) -> TrainingPipeline:
        info = self._training_pipeline_info()

        return TrainingPipeline(
            gamma=info.get("gamma", 0.99),
            use_gae=info.get("use_gae", True),
            gae_lambda=info.get("gae_lambda", 0.95),
            num_steps=info["num_steps"],
            num_mini_batch=info["num_mini_batch"],
            update_repeats=info["update_repeats"],
            max_grad_norm=info.get("max_grad_norm", 0.5),
            save_interval=self.SAVE_INTERVAL,
            named_losses=info["named_losses"],
            metric_accumulate_interval=self.num_train_processes()
            * max(*self.MAX_STEPS.values())
            if torch.cuda.is_available()
            else 1,
            optimizer_builder=Builder(optim.Adam, dict(lr=info["lr"])),
            advance_scene_rollout_period=None,
            pipeline_stages=info["pipeline_stages"],
            lr_scheduler_builder=self.get_lr_scheduler_builder(
                use_lr_decay=info["use_lr_decay"]
            ),
        )

    def create_model(self, **kwargs) -> nn.Module:
        if self.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None:
            raise NotImplementedError
        else:
            return ResNetRearrangeActorCriticRNN(
                action_space=gym.spaces.Discrete(len(self.actions())),
                observation_space=kwargs[
                    "sensor_preprocessor_graph"
                ].observation_spaces,
                rgb_uuid=self.EGOCENTRIC_RGB_RESNET_UUID,
                unshuffled_rgb_uuid=self.UNSHUFFLED_RGB_RESNET_UUID,
                num_rnn_layers=self.GRU_LAYERS,
            )
