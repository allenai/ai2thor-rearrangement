import math
import os
import random
from typing import Optional, Dict, Any
import sys
import json

import compress_pickle
from invoke import task

from rearrange.procthor_rearrange.environment import RearrangeProcTHOREnvironment


def make_env(**kwargs):
    assert ("height" in kwargs) == ("width" in kwargs)

    controller_kwargs = {
        **kwargs,
        "scene": "Procedural",
    }

    return RearrangeProcTHOREnvironment(
        force_cache_reset=True, controller_kwargs=controller_kwargs,
    )


def load_procthor_rearrange(mode="train", folder="data/2022procthor"):
    import os
    import compress_pickle

    path = os.path.join(folder, f"{mode}.pkl.gz")
    assert os.path.exists(path), f"missing path {path}"

    return compress_pickle.load(path)


@task
def consolidate_train_dataset(ctx, output_name="train_consolidated"):
    import os
    import compress_pickle

    from datagen.procthor_datagen.datagen_runner import STARTER_DATA_DIR

    path = os.path.join(STARTER_DATA_DIR, f"train.pkl.gz")
    if os.path.exists(path):
        generated_episodes = compress_pickle.load(path)
    else:
        print(f"Missing {path}. DONE.")
        return

    filtered_episodes = {}
    total_episodes = 0
    for scene, specs in generated_episodes.items():
        valid_specs = [spec for spec in specs if spec is not None and spec != -1]
        if len(valid_specs) > 0:
            filtered_episodes[scene] = valid_specs
            total_episodes += len(valid_specs)

    consolidated_path = os.path.join(STARTER_DATA_DIR, f"{output_name}.pkl.gz")
    print(
        f"{consolidated_path} contains {len(filtered_episodes)} scenes with a total of {total_episodes} episodes"
        f" ({total_episodes / len(filtered_episodes):.2f} episodes per scene)"
    )

    compress_pickle.dump(
        obj=filtered_episodes,
        path=consolidated_path,
        pickler_kwargs={"protocol": 4},  # Backwards compatible with python 3.6
    )

    split_training_data(ctx, output_name)

    print("DONE")


@task
def split_training_data(ctx, input_name="train_consolidated"):
    return split_data(ctx, mode="train", input_name=input_name)


@task
def make_ithor_mini_val(ctx):
    import compress_pickle
    import os

    val = load_procthor_rearrange("val", "data/2022")

    mini_val = {}
    num_episodes = 0
    for scene in val:
        mini_val[scene] = val[scene][::5]
        num_episodes += len(mini_val[scene])

    print(
        f"{len(mini_val)} scenes, {num_episodes} episodes ({num_episodes/len(mini_val)} per scene)"
    )

    os.makedirs("data/2022", exist_ok=True)
    compress_pickle.dump(
        obj=mini_val,
        path="data/2022/ithor_mini_val.pkl.gz",
        pickler_kwargs={"protocol": 4},  # Backwards compatible with python 3.6
    )

    print("DONE")


@task
def consolidate_procthor_val(ctx, output_name="val_consolidated"):
    import os
    import compress_pickle

    from datagen.procthor_datagen.datagen_runner import STARTER_DATA_DIR

    random.seed(123456)

    path = os.path.join(STARTER_DATA_DIR, f"val.pkl.gz")
    if os.path.exists(path):
        generated_episodes = compress_pickle.load(path)
    else:
        print(f"Missing {path}. DONE.")
        return

    env = make_env()

    filtered_episodes = {}
    total_episodes = 0
    scenes = list(generated_episodes.keys())
    num_rooms = []
    for scene in scenes:
        specs = generated_episodes[scene]
        valid_specs = [spec for spec in specs if spec is not None and spec != -1]
        if len(valid_specs) != 10:
            print(f"skipped missing episodes {len(valid_specs)}")
            continue

        num_openables = len(
            [spec for spec in valid_specs if len(spec["openable_data"]) > 0]
        )
        if num_openables != 5:
            print("skipped missing openables")
            continue

        num_rooms.append(env.num_rooms(scene))

        filtered_episodes[scene] = valid_specs
        total_episodes += len(valid_specs)

    consolidated_path = os.path.join(STARTER_DATA_DIR, f"{output_name}.pkl.gz")
    print(
        f"{consolidated_path} contains {len(filtered_episodes)} scenes with a total of {total_episodes} episodes"
        f" ({total_episodes / len(filtered_episodes):.2f} episodes per scene)"
    )

    compress_pickle.dump(
        obj=filtered_episodes,
        path=consolidated_path,
        pickler_kwargs={"protocol": 4},  # Backwards compatible with python 3.6
    )

    print(num_rooms)

    print("DONE")


@task
def split_data(ctx, mode="val", input_name="val_consolidated"):
    from rearrange.procthor_rearrange.constants import STARTER_DATA_DIR
    import compress_pickle

    all_data = load_procthor_rearrange(input_name)

    all_idxs = sorted([int(scene.split("_")[-1]) for scene in list(all_data.keys())])

    output_folder = os.path.join(
        STARTER_DATA_DIR, f"split_{input_name.replace('_consolidated', '')}"
    )
    os.makedirs(output_folder, exist_ok=True)

    # Make groups of up to 100 scenes (for small houses) or 400 scenes (all 10k houses)
    group = 100 if len(all_idxs) < 3000 else 400
    for first_idx in range(0, len(all_idxs), group):
        last_idx = min(len(all_idxs), first_idx + group)
        print(f"{all_idxs[first_idx]} to {all_idxs[last_idx - 1]}")
        current_keys = [f"{mode}_{all_idxs[idx]}" for idx in range(first_idx, last_idx)]
        partial_dataset = {key: all_data[key] for key in current_keys}

        consolidated_path = os.path.join(
            output_folder,
            f"{mode}_{all_idxs[first_idx]}_{all_idxs[last_idx - 1]}.pkl.gz",
        )
        compress_pickle.dump(
            obj=partial_dataset,
            path=consolidated_path,
            pickler_kwargs={"protocol": 4},  # Backwards compatible with python 3.6
        )

    print("DONE")


@task
def make_procthor_mini_val(
    ctx,
    stats_file="data/2022procthor/procthor_valid_scene_stats.json",
    chosen_scenes=20,
    seed=12345,
    num_attempts=10,
):
    from collections import defaultdict
    import compress_pickle
    import os
    import heapq as hq
    import numpy as np

    random.seed(seed)

    val = load_procthor_rearrange("val_consolidated", "data/2022procthor")

    if os.path.isfile(stats_file):
        with open(stats_file, "r") as f:
            stats = json.load(f)
    else:
        stats = {}

    if len(stats) < len(val):
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=4, sort_keys=True)

        env = make_env()

        for scene in val:
            if scene in stats:
                continue

            print(scene)
            env.procthor_reset(scene)

            target_freqs = defaultdict(int)

            id_to_type = {obj["objectId"]: obj["objectType"] for obj in env.objects()}
            name_to_type = {obj["name"]: obj["objectType"] for obj in env.objects()}

            expected_targets = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
            for epit, ep in enumerate(val[scene]):
                num_targets = 0

                if len(ep["openable_data"]) > 0:
                    target_freqs[id_to_type[ep["openable_data"][0]["objectId"]]] += 1
                    num_targets += 1

                for sp, tp in zip(ep["starting_poses"], ep["target_poses"]):
                    assert sp["name"] == tp["name"]

                    if (
                        math.sqrt(
                            sum(
                                (sp["position"][x] - tp["position"][x]) ** 2
                                for x in "xyz"
                            )
                        )
                        > 0.01
                    ):
                        target_freqs[name_to_type[sp["name"]]] += 1
                        num_targets += 1

                if num_targets != expected_targets[epit]:
                    print(
                        f"{scene} {epit} had {num_targets} for expected {expected_targets[epit]}"
                    )

            stats[scene] = {**target_freqs}

            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=4, sort_keys=True)

        env.stop()

    scenes = list(stats.keys())

    max_length = 0
    best_set = None
    min_ratio = float(np.inf)
    for attempt in range(num_attempts):
        random.shuffle(scenes)

        total_target_freqs = defaultdict(int)
        pqueue = []
        for scene in scenes:
            target_freqs = stats[scene]

            scene_priority = -np.sum(
                [1.0 / (1e-6 + total_target_freqs[typ]) for typ in target_freqs]
            )

            hq.heappush(pqueue, (scene_priority, scene))

            # Update target type frequencies
            for typ in target_freqs:
                total_target_freqs[typ] += target_freqs[typ]

        mini_val = {}
        num_episodes = 0
        included_targets = defaultdict(int)
        for it in range(chosen_scenes):
            scene = hq.heappop(pqueue)[1]
            mini_val[scene] = val[scene]
            num_episodes += len(mini_val[scene])
            target_freqs = stats[scene]
            for typ in target_freqs:
                included_targets[typ] += target_freqs[typ]

        if len(included_targets) < max_length:
            continue

        if max(included_targets.values()) / min(included_targets.values()) >= min_ratio:
            if len(included_targets) == max_length:
                continue

        max_length = len(included_targets)
        min_ratio = max(included_targets.values()) / min(included_targets.values())
        best_set = mini_val

        print(
            f"Attempt {attempt} {len(mini_val)} scenes, {num_episodes} episodes ({num_episodes/len(mini_val)} per scene)"
        )

        # included targets
        included_targets = sorted(
            [(included_targets[typ], typ) for typ in included_targets], reverse=True
        )
        print(
            f"Attempt {attempt} Included targets ({len(included_targets)}, ratio {min_ratio}): {included_targets}"
        )

    included_targets = defaultdict(int)
    random.shuffle(scenes)
    for it in range(chosen_scenes):
        target_freqs = stats[scenes[it]]
        for typ in target_freqs:
            included_targets[typ] += target_freqs[typ]
    included_targets = sorted(
        [(included_targets[typ], typ) for typ in included_targets], reverse=True
    )
    print(f"Random choice targets ({len(included_targets)}): {included_targets}")

    os.makedirs("data/2022", exist_ok=True)
    compress_pickle.dump(
        obj=best_set,
        path="data/2022procthor/mini_val_consolidated.pkl.gz",
        pickler_kwargs={"protocol": 4},  # Backwards compatible with python 3.6
    )

    split_data(ctx, mode="val", input_name="mini_val_consolidated")

    print("DONE")


@task
def make_valid_houses_file(
    ctx, num_valid_houses=1_000, prefix="mini_val", verbose=True
):
    used_houses = [None] * num_valid_houses  # Assume
    from utils.procthor_utils import Houses

    houses = Houses()
    houses.mode("validation")
    assert len(houses) == num_valid_houses

    episodes = compress_pickle.load(
        os.path.join("data", "2022procthor", f"{prefix}_consolidated.pkl.gz")
    )
    num_used = 0
    for scene in episodes:
        if verbose:
            print(scene)
        pos = int(scene.split("_")[-1])
        used_houses[pos] = houses._data[houses._mode][pos]
        num_used += 1

    ofilename = os.path.join("data", "2022procthor", f"{prefix}_houses.pkl.gz")
    print(f"Writing {num_used} houses to {ofilename}")

    compress_pickle.dump(used_houses, ofilename)

    if verbose:
        print("DONE")


@task
def make_procthor_mini_train(
    ctx,
    stats_file="data/2022procthor/procthor_train_scene_stats.json",
    chosen_scenes=2500,
    seed=12345,
    num_attempts=10,
):
    from collections import defaultdict
    import compress_pickle
    import os
    import heapq as hq
    import numpy as np
    from utils.multiprocessing_utils import Manager, Worker

    random.seed(seed)

    class LazyEnv:
        def __init__(self, **kwargs):
            self._env = None
            self._kwargs = kwargs

        def stop(self):
            try:
                if self._env is not None:
                    self._env.stop()
            finally:
                self._env = None

        def _ensure_env(self):
            if self._env is None:
                self._env = make_env(**self._kwargs)

        def procthor_reset(self, scene):
            self._ensure_env()
            return self._env.procthor_reset(scene)

        def objects(self):
            self._ensure_env()
            return self._env.objects()

    class StatsWorker(Worker):
        def create_env(self, **env_args: Any):
            if self.env is not None:
                try:
                    self.env.stop()
                except:
                    print("Failed stop")
            if self.gpu is not None:
                return LazyEnv(x_display=f"0.{self.gpu}")
            else:
                return LazyEnv()

        def work(
            self, task_type: Optional[str], task_info: Dict[str, Any]
        ) -> Optional[Any]:
            scene = task_info["scene"]
            if scene is None:
                return None

            self.env.procthor_reset(scene)
            id_to_type = {
                obj["objectId"]: obj["objectType"] for obj in self.env.objects()
            }
            name_to_type = {
                obj["name"]: obj["objectType"] for obj in self.env.objects()
            }
            return dict(id_to_type=id_to_type, name_to_type=name_to_type)

    class StatsManager(Manager):
        def save(self):
            import time

            ctime = time.time()
            if (
                not hasattr(self, "last_save_time")
                or (ctime - self.last_save_time) > 20 * 60
                or self.all_work_done
            ):
                with open(stats_file, "w") as f:
                    json.dump(self.stats, f, indent=4, sort_keys=True)
                self.last_save_time = time.time()
                print(f"Took {self.last_save_time - ctime} s to save")

        def load(self):
            if os.path.isfile(stats_file):
                try:
                    with open(stats_file, "r") as f:
                        self.stats = json.load(f)
                except:
                    self.stats = {}
            else:
                self.stats = {}

        def work(
            self,
            task_type: Optional[str],
            task_info: Dict[str, Any],
            success: bool,
            result: Any,
        ) -> None:
            if self.first_tasks_needed:
                self.raw = load_procthor_rearrange("train", "data/2022procthor")
                print(f"Loaded {len(self.raw)}")

                self.load()

                self.enqueue(dict(scene=None))

                if len(self.stats) < len(self.raw):
                    self.save()

                    for scene in self.raw:
                        if scene in self.stats:
                            continue

                        expected_targets = sum([[it + 1] * 4 for it in range(5)], [])
                        as_expected = True

                        if any(ep is None for ep in self.raw[scene]):
                            self.stats[scene] = None
                            continue
                        elif any(ep == -1 for ep in self.raw[scene]):
                            continue

                        for epit, ep in enumerate(self.raw[scene]):
                            num_targets = 0

                            if len(ep["openable_data"]) > 0:
                                num_targets += 1

                            for sp, tp in zip(ep["starting_poses"], ep["target_poses"]):
                                assert sp["name"] == tp["name"]

                                if (
                                    math.sqrt(
                                        sum(
                                            (sp["position"][x] - tp["position"][x]) ** 2
                                            for x in "xyz"
                                        )
                                    )
                                    > 0.01
                                ):
                                    num_targets += 1

                            if num_targets != expected_targets[epit]:
                                print(
                                    f"{scene} {epit} had {num_targets} for expected {expected_targets[epit]}. Skip."
                                )
                                as_expected = False
                                break

                        if as_expected:
                            self.enqueue(dict(scene=scene))
                        else:
                            self.stats[scene] = None
            else:
                if result is None:
                    return
                id_to_type, name_to_type = result["id_to_type"], result["name_to_type"]
                scene = task_info["scene"]

                target_freqs = defaultdict(int)

                for epit, ep in enumerate(self.raw[scene]):
                    if len(ep["openable_data"]) > 0:
                        target_freqs[
                            id_to_type[ep["openable_data"][0]["objectId"]]
                        ] += 1

                    for sp, tp in zip(ep["starting_poses"], ep["target_poses"]):
                        assert sp["name"] == tp["name"]
                        if (
                            math.sqrt(
                                sum(
                                    (sp["position"][x] - tp["position"][x]) ** 2
                                    for x in "xyz"
                                )
                            )
                            > 0.01
                        ):
                            target_freqs[name_to_type[sp["name"]]] += 1

                self.stats[scene] = {**target_freqs}
                self.save()

    import torch
    import multiprocessing as mp

    stats_manager = StatsManager(
        StatsWorker,
        workers=max((3 * mp.cpu_count()) // 4, 1),
        ngpus=torch.cuda.device_count(),
        debugging=False,
    )

    random.seed(seed)

    stats = stats_manager.stats

    scenes = [scene for scene in stats if stats[scene] is not None]
    print(f"{len(scenes)} available scenes")

    max_length = 0
    best_set = None
    min_ratio = float(np.inf)
    for attempt in range(num_attempts):
        random.shuffle(scenes)

        total_target_freqs = defaultdict(int)
        pqueue = []
        for scene in scenes:
            target_freqs = stats[scene]

            scene_priority = -np.sum(
                [1.0 / (1e-6 + total_target_freqs[typ]) for typ in target_freqs]
            )

            hq.heappush(pqueue, (scene_priority, scene))

            # Update target type frequencies
            for typ in target_freqs:
                total_target_freqs[typ] += target_freqs[typ]

        mini_val = {}
        num_episodes = 0
        included_targets = defaultdict(int)
        for it in range(min(chosen_scenes, len(scenes))):
            scene = hq.heappop(pqueue)[1]
            mini_val[scene] = stats_manager.raw[scene]
            num_episodes += len(mini_val[scene])
            target_freqs = stats[scene]
            for typ in target_freqs:
                included_targets[typ] += target_freqs[typ]

        if len(included_targets) < max_length:
            continue

        if max(included_targets.values()) / min(included_targets.values()) >= min_ratio:
            if len(included_targets) == max_length:
                continue

        max_length = len(included_targets)
        min_ratio = max(included_targets.values()) / min(included_targets.values())
        best_set = mini_val

        print(
            f"Attempt {attempt} {len(mini_val)} scenes, {num_episodes} episodes ({num_episodes/len(mini_val)} per scene)"
        )

        # included targets
        included_targets = sorted(
            [(included_targets[typ], typ) for typ in included_targets], reverse=True
        )
        print(
            f"Attempt {attempt} Included targets ({len(included_targets)}, ratio {min_ratio}): {included_targets}"
        )

    included_targets = defaultdict(int)
    random.shuffle(scenes)
    for it in range(min(chosen_scenes, len(scenes))):
        target_freqs = stats[scenes[it]]
        for typ in target_freqs:
            included_targets[typ] += target_freqs[typ]
    included_targets = sorted(
        [(included_targets[typ], typ) for typ in included_targets], reverse=True
    )
    print(f"Random choice targets ({len(included_targets)}): {included_targets}")

    os.makedirs("data/2022procthor", exist_ok=True)
    compress_pickle.dump(
        obj=best_set,
        path="data/2022procthor/train_consolidated.pkl.gz",
        pickler_kwargs={"protocol": 4},  # Backwards compatible with python 3.6
    )

    split_data(ctx, mode="train", input_name="train_consolidated")

    print("DONE")


@task
def install_procthor_dataset(ctx, revision="2022procthor", skip_consolidate=False):
    import prior
    from rearrange_constants import ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR

    all_data = prior.load_dataset("rearrangement_episodes", revision=revision)

    for partition in ["val", "train"]:
        output_partition = f"mini_{partition}" if partition in ["val"] else partition

        print(f"{output_partition}...")

        num_episodes = 0

        current_dir = os.path.join(
            ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR,
            "data",
            "2022procthor",
            f"split_{output_partition}",
        )
        os.makedirs(current_dir, exist_ok=True)

        consolidated_data = {}

        for part, compressed_part_data in all_data[partition]:
            print(f"{part}")

            if not skip_consolidate:
                # each part is a compressed_pickle
                cur_data = compress_pickle.loads(
                    data=compressed_part_data, compression="gzip"
                )

                for scene in cur_data:
                    num_episodes += len(cur_data[scene])

                consolidated_data.update(cur_data)

            with open(os.path.join(current_dir, f"{part}.pkl.gz"), "wb") as f:
                f.write(compressed_part_data)

        if not skip_consolidate:
            print(f"{output_partition}_consolidated")
            consolidated_file = os.path.join(
                ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR,
                "data",
                "2022procthor",
                f"{output_partition}_consolidated.pkl.gz",
            )
            compress_pickle.dump(
                obj=consolidated_data,
                path=consolidated_file,
                pickler_kwargs={"protocol": 4},  # Backwards compatible with python 3.6
            )

            print(
                f"{len(consolidated_data)} scenes and total {num_episodes} episodes for {output_partition}"
            )

    print("Creating mini val houses file")
    make_valid_houses_file(ctx, verbose=False)

    print("DONE")


@task
def install_ithor_dataset(ctx, data_subdir="2022", revision=None):
    import prior
    from rearrange_constants import ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR

    if revision is None:
        revision = f"{data_subdir}ithor"

    all_data = prior.load_dataset("rearrangement_episodes", revision=revision)

    current_dir = os.path.join(
        ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR, "data", data_subdir,
    )
    os.makedirs(current_dir, exist_ok=True)

    for part, compressed_part_data in all_data["test"]:
        with open(os.path.join(current_dir, f"{part}.pkl.gz"), "wb") as f:
            f.write(compressed_part_data)

    print("DONE")
