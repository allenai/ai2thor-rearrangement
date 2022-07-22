"""Include the Task and TaskSampler to train on a single unshuffle instance."""
import copy
import glob
import os
import random
import traceback
from abc import ABC
from typing import Any, Tuple, Optional, Dict, Sequence, List, Union, cast, Set

import compress_pickle
import gym.spaces
import numpy as np
import stringcase

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import SensorSuite
from allenact.base_abstractions.task import Task, TaskSampler
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_util import round_to_factor
from allenact.utils.misc_utils import prepare_locals_for_super

from rearrange.procthor_rearrange.constants import STARTER_DATA_DIR
from rearrange.procthor_rearrange.environment import RearrangeProcTHOREnvironment
from rearrange.procthor_rearrange.expert import (
    GreedyExploreUnshuffleExpert,
    ShortestPathNavigatorProcTHOR,
)
from rearrange.constants import STEP_SIZE
from rearrange.environment import RearrangeTaskSpec
from rearrange.tasks import (
    WalkthroughTask as BaseWalkthroughTask,
    UnshuffleTask as BaseUnshuffleTask,
    RearrangeTaskSpecIterable as BaseRearrangeTaskSpecIterable,
    RearrangeTaskSampler as BaseRearrangeTaskSampler,
)
from rearrange.utils import (
    RearrangeActionSpace,
    include_object_data,
)

from rearrange.tasks import AbstractRearrangeTask


class UnshuffleTask(BaseUnshuffleTask):
    NAME_KEY = "objectId"

    def __init__(
        self,
        sensors: SensorSuite,
        unshuffle_env: RearrangeProcTHOREnvironment,
        walkthrough_env: RearrangeProcTHOREnvironment,
        max_steps: int,
        discrete_actions: Tuple[str, ...],
        require_done_action: bool = False,
        locations_visited_in_walkthrough: Optional[np.ndarray] = None,
        object_names_seen_in_walkthrough: Set[str] = None,
        metrics_from_walkthrough: Optional[Dict[str, Any]] = None,
        task_spec_in_metrics: bool = False,
        expert_exploration_enabled: bool = True,
    ) -> None:
        super_init_args = prepare_locals_for_super(locals())
        super_init_args.pop("expert_exploration_enabled")
        super().__init__(**super_init_args)
        self.expert_exploration_enabled = expert_exploration_enabled

    def create_navigator(self):
        return ShortestPathNavigatorProcTHOR(
            env=self.unshuffle_env,
            grid_size=STEP_SIZE,
            include_move_left_right=all(
                f"move_{k}" in self.action_names() for k in ["left", "right"]
            ),
        )

    def create_expert(self):
        return GreedyExploreUnshuffleExpert(
            task=self,
            shortest_path_navigator=self.unshuffle_env.shortest_path_navigator,
            exploration_enabled=self.expert_exploration_enabled,
        )

    @property
    def expert_priority(self):
        return self.greedy_expert.object_id_to_priority


class RearrangeTaskSpecIterable(BaseRearrangeTaskSpecIterable):
    """Iterate through a collection of scenes and pose specifications for the
    rearrange task."""

    def preprocess_spec_dict(self, spec_dict):
        return compress_pickle.loads(spec_dict, compression="gzip")


class RearrangeTaskSampler(BaseRearrangeTaskSampler):
    def __init__(
        self,
        run_walkthrough_phase: bool,
        run_unshuffle_phase: bool,
        stage: str,
        scenes_to_task_spec_dicts: Dict[str, List[Dict[str, Any]]],
        rearrange_env_kwargs: Optional[Dict[str, Any]],
        sensors: SensorSuite,
        max_steps: Union[Dict[str, int], int],
        discrete_actions: Tuple[str, ...],
        require_done_action: bool,
        force_axis_aligned_start: bool,
        epochs: Union[int, float, str] = "default",
        seed: Optional[int] = None,
        unshuffle_runs_per_walkthrough: Optional[int] = None,
        task_spec_in_metrics: bool = False,
        expert_exploration_enabled: bool = True,
    ) -> None:
        super_init_args = prepare_locals_for_super(locals())
        super_init_args.pop("expert_exploration_enabled")
        super().__init__(**super_init_args)
        self.expert_exploration_enabled = expert_exploration_enabled

    def make_task_spec_iterable(self, epochs):
        return RearrangeTaskSpecIterable(
            scenes_to_task_spec_dicts=self.scenes_to_task_spec_dicts,
            seed=self.main_seed,
            epochs=epochs,
            shuffle=epochs == float("inf"),
        )

    def create_env(self, **kwargs):
        return RearrangeProcTHOREnvironment(**kwargs)

    @classmethod
    def get_base_dir(cls):
        return STARTER_DATA_DIR

    @classmethod
    def load_rearrange_data_from_path(
        cls,
        stage: str,
        base_dir: Optional[str] = None,
        scenes: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        stage = stage.lower()

        if stage in ["valid", "val"]:
            stage = "val"
            folder = "split_val"
        elif stage in ["train", "training"]:
            folder = "split_train"
        elif stage in ["mini_val", "mini_valid"]:
            stage = "val"
            folder = "split_mini_val"
        else:
            folder = "unknown"

        if scenes is not None:
            idxs_scenes = sorted(
                [(int(scene.split("_")[-1]), scene) for scene in scenes]
            )
            firsts_lasts_fnames = []

            for fname in glob.glob(os.path.join(base_dir, folder, "*.pkl.gz")):
                vals = fname.replace(".pkl.gz", "").split("_")[-2:]
                firsts_lasts_fnames.append((int(vals[0]), int(vals[1]), fname))
            firsts_lasts_fnames = sorted(firsts_lasts_fnames)

            from collections import defaultdict

            fname_to_scenes = defaultdict(list)
            f = 0
            for idx, scene in idxs_scenes:
                while firsts_lasts_fnames[f][1] < idx:
                    f += 1
                assert firsts_lasts_fnames[f][0] <= idx <= firsts_lasts_fnames[f][1]
                fname_to_scenes[firsts_lasts_fnames[f][2]].append(scene)

            data = {}
            for fname, scenes in fname_to_scenes.items():
                get_logger().info(f"Loading data from {fname}")
                partial_data = compress_pickle.load(path=fname)
                for scene in scenes:
                    data[scene] = partial_data[scene]
        else:
            data_path = os.path.abspath(os.path.join(base_dir, f"{stage}.pkl.gz"))
            if not os.path.exists(data_path):
                raise RuntimeError(f"No data at path {data_path}")

            get_logger().info(f"Loading data from {data_path}")
            data = compress_pickle.load(path=data_path)

        for scene in data:
            for ind, task_spec_dict in enumerate(data[scene]):
                task_spec_dict["scene"] = scene

                if "index" not in task_spec_dict:
                    task_spec_dict["index"] = ind

                if "stage" not in task_spec_dict:
                    task_spec_dict["stage"] = stage

                data[scene][ind] = compress_pickle.dumps(
                    task_spec_dict, compression="gzip"
                )
        return data

    def walkthrough_env_post_reset(self):
        self.walkthrough_env.controller.step(action="SetObjectFilter", objectIds=[])

    def create_unshuffle_task(self):
        return UnshuffleTask(
            sensors=self.sensors,
            unshuffle_env=self.unshuffle_env,
            walkthrough_env=self.walkthrough_env,
            max_steps=self.max_steps["unshuffle"],
            discrete_actions=self.discrete_actions,
            require_done_action=self.require_done_action,
            task_spec_in_metrics=self.task_spec_in_metrics,
            expert_exploration_enabled=self.expert_exploration_enabled,
        )

    def create_unshuffle_after_walkthrough_task(self, walkthrough_task):
        return UnshuffleTask(
            sensors=self.sensors,
            unshuffle_env=self.unshuffle_env,
            walkthrough_env=self.walkthrough_env,
            max_steps=self.max_steps["unshuffle"],
            discrete_actions=self.discrete_actions,
            require_done_action=self.require_done_action,
            locations_visited_in_walkthrough=np.array(
                tuple(walkthrough_task.visited_positions_xzrsh)
            ),
            object_names_seen_in_walkthrough=copy.copy(
                walkthrough_task.seen_pickupable_objects
                | walkthrough_task.seen_openable_objects
            ),
            metrics_from_walkthrough=walkthrough_task.metrics(force_return=True),
            task_spec_in_metrics=self.task_spec_in_metrics,
            expert_exploration_enabled=False,
        )
