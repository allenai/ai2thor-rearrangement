from baseline_configs.one_phase.procthor.one_phase_rgb_clip_dagger import (
    ProcThorOnePhaseRGBClipResNet50DaggerTrainMultiNodeConfig as BaseConfig,
)

import os
import copy
from typing import Optional, Sequence, Dict
import glob

import compress_json
import compress_pickle

from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.utils.system import get_logger

from rearrange_constants import ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR
from rearrange.procthor_rearrange.tasks import RearrangeTaskSampler
from baseline_configs.one_phase.procthor.eval.eval_minivalid_ithor import (
    EvalLoaderMixin as BaseMixin,
)


class EvalLoaderMixin(BaseMixin):
    DATA_PREFIX = "split_"
    SAMPLER_STAGE = "mini_val"

    @staticmethod
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
    EVAL_MIXIN = EvalLoaderMixin

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

        return dict(
            force_cache_reset=True,
            epochs=1,
            task_spec_in_metrics=task_spec_in_metrics,
            **self.EVAL_MIXIN.prepare_stagewise_task_sampler_args(
                process_ind=process_ind,
                total_processes=total_processes,
                sensors=copy.deepcopy(self.sensors()),
                devices=devices,
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
