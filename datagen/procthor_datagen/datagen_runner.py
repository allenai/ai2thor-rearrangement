"""A script for generating rearrangement datasets."""

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import platform
import random
import warnings
from collections import defaultdict
from typing import List, Set, Dict, Optional, Any, cast
import time

import compress_pickle
import numpy as np
import torch.cuda
import tqdm
from allenact.utils.misc_utils import md5_hash_str_as_int
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment

from datagen.datagen_constants import OBJECT_TYPES_TO_NOT_MOVE
from rearrange_constants import OPENNESS_THRESHOLD, IOU_THRESHOLD

from datagen.procthor_datagen.datagen_utils import (
    get_random_seeds,
    filter_pickupable,
    open_objs,
    remove_objects_until_all_have_identical_meshes,
    check_object_opens,
    mapping_counts,
)
from utils.multiprocessing_utils import Manager, Worker, get_logger
from rearrange.procthor_rearrange.constants import STARTER_DATA_DIR
from rearrange.procthor_rearrange.environment import (
    RearrangeProcTHOREnvironment,
    RearrangeTaskSpec,
)
from rearrange.utils import extract_obj_data

NUM_TRAIN_UNSEEN_EPISODES = 1_000  # 1 episode per scene
NUM_TRAIN_SCENES = 10_000  # N episodes per scene
NUM_VALID_SCENES = 1_000  # 10 episodes per scene
NUM_TEST_SCENES = 1_000  # 1 episode per scene

MAX_ROOMS_IN_HOUSE = 2

MAX_POS_IN_HOUSE = 700

MAX_TRIES = 40
EXTENDED_TRIES = 10

mp = mp.get_context("spawn")

# Includes types used in both open and pickup actions:
VALID_TARGET_TYPES = {
    "AlarmClock",
    "AluminumFoil",
    "Apple",
    "BaseballBat",
    "BasketBall",
    "Blinds",
    "Book",
    "Boots",
    "Bottle",
    "Bowl",
    "Box",
    "Bread",
    "ButterKnife",
    "CD",
    "Cabinet",
    "Candle",
    "CellPhone",
    "Cloth",
    "CreditCard",
    "Cup",
    "DishSponge",
    "Drawer",
    "Dumbbell",
    "Egg",
    "Footstool",
    "Fork",
    "Fridge",
    "HandTowel",
    "Kettle",
    "KeyChain",
    "Knife",
    "Ladle",
    "Laptop",
    "LaundryHamper",
    "Lettuce",
    "Microwave",
    "Mug",
    "Newspaper",
    "Pan",
    "PaperTowelRoll",
    "Pen",
    "Pencil",
    "PepperShaker",
    "Pillow",
    "Plate",
    "Plunger",
    "Pot",
    "Potato",
    "RemoteControl",
    "Safe",
    "SaltShaker",
    "ScrubBrush",
    "ShowerCurtain",
    "ShowerDoor",
    "SoapBar",
    "SoapBottle",
    "Spatula",
    "Spoon",
    "SprayBottle",
    "Statue",
    "TableTopDecor",
    "TeddyBear",
    "TennisRacket",
    "TissueBox",
    "Toilet",
    "ToiletPaper",
    "Tomato",
    "Towel",
    "Vase",
    "Watch",
    "WateringCan",
    "WineBottle",
}


def get_scene_limits(
    env: RearrangeProcTHOREnvironment, scene: str, object_types_to_not_move: Set[str],
):
    controller = env.controller

    if not env.procthor_reset(scene_name=scene, force_reset=True):
        print(f"Cannot reset scene {scene}")
        return None

    if not remove_objects_until_all_have_identical_meshes(controller):
        print(f"Failed to remove_objects_until_all_have_identical_meshes in {scene}")
        return None

    all_objects = controller.last_event.metadata["objects"]
    if any(o["isBroken"] for o in all_objects):
        print(f"broken objects in {scene}")
        return None

    room_reachable, reachability_meta = env.all_rooms_reachable()
    if not room_reachable:
        print(f"Unreachable rooms in {scene}: {reachability_meta}")
        return None

    openable_objects = env.obj_id_with_cond_to_room(
        lambda o: o["openable"]
        and not o["pickupable"]
        and o["objectType"] in VALID_TARGET_TYPES
    )

    if len(openable_objects) == 0:
        print(f"No objects to open in {scene}")
        return None

    meta_rps = controller.step("GetReachablePositions").metadata
    if meta_rps["lastActionSuccess"]:
        rps = meta_rps["actionReturn"][:]
    else:
        print(
            f"In {scene}, couldn't get reachable positions despite all rooms being reachable (?)"
        )
        return None

    if len(rps) > MAX_POS_IN_HOUSE:
        print(f"{scene} has more than {MAX_POS_IN_HOUSE} reachable positions.")
        return None

    all_objects = env.ids_to_objs()

    room_to_openable_ids = defaultdict(list)
    for oid, room in openable_objects.items():
        interactable_poses = env.controller.step(
            "GetInteractablePoses", objectId=oid, positions=rps,
        ).metadata["actionReturn"]
        if interactable_poses is None or len(interactable_poses) == 0:
            continue

        could_open_close, could_open, could_close = check_object_opens(
            all_objects[oid], controller, return_open_closed=True
        )
        if not could_close:
            if could_open:
                print(f"Couldn't close {oid} fully in {scene}")
                return None
            continue
        if could_open_close:
            room_to_openable_ids[room].append(oid)

    pickupable_objects = filter_pickupable(
        objects=[
            all_objects[obj]
            for obj in all_objects
            if all_objects[obj]["objectType"] in VALID_TARGET_TYPES
        ],
        object_types_to_not_move=object_types_to_not_move,
    )

    if len(room_to_openable_ids) + len(pickupable_objects) == 0:
        print(f"No objects to open or pickup in {scene}")
        return None

    if len(pickupable_objects) < len(env.room_to_poly) * 5:
        print(f"Less than 5 pickupables per room in {scene}")
        return None

    # Does not take into account reachability, receptability
    receps_per_room = {
        room: env.static_receptacles_in_room(room) for room in env.room_to_poly
    }

    for room, rids in receps_per_room.items():
        reachable_ids = []
        for rid in rids:
            interactable_poses = env.controller.step(
                "GetInteractablePoses", objectId=rid, positions=rps,
            ).metadata["actionReturn"]
            if interactable_poses is None or len(interactable_poses) == 0:
                continue
            else:
                reachable_ids.append(rid)
        receps_per_room[room] = reachable_ids

    num_receps_per_room = mapping_counts(receps_per_room)

    if any(v < 2 for v in num_receps_per_room.values()):
        print(
            f"Less than 2 receptacles in some room(s) in {scene}: {num_receps_per_room}"
        )
        return None

    return dict(
        room_openables={**room_to_openable_ids},  # in each room
        max_pickupable=len(pickupable_objects),  # in the entire house
        max_receps=num_receps_per_room,
    )


def try_to_open(
    env, single_room, objects_after_first_irs, num_objs_to_open, possible_openable_ids
):
    # Of the non-movable objects, randomly open some of them before shuffling other objects
    openable_objects = [
        obj
        for obj in objects_after_first_irs
        if obj["openable"]
        and not obj["pickupable"]
        and obj["objectType"] in VALID_TARGET_TYPES
    ]
    random.shuffle(openable_objects)

    openable_id_to_room = env.obj_id_with_cond_to_room(source=openable_objects)

    object_ids_to_open = []
    for oo in openable_objects:
        if len(object_ids_to_open) == num_objs_to_open:
            break
        if oo["objectId"] not in possible_openable_ids:
            continue
        if openable_id_to_room[oo["objectId"]] != single_room:
            continue
        if check_object_opens(oo, env.controller):
            object_ids_to_open.append(oo["objectId"])

    if len(object_ids_to_open) != num_objs_to_open:
        print("Insufficient number of objects to open")
        return None, None

    try:
        start_openness = open_objs(
            object_ids_to_open=object_ids_to_open, controller=env.controller
        )
    except (StopIteration, RuntimeError):
        print("Failed opening")
        return None, None

    return object_ids_to_open, start_openness


def group_pickupables(
    env,
    objects_after_first_irs,
    object_types_to_not_move,
    single_room,
    num_objs_to_move,
):
    valid_pickupables_after_first_irs = [
        obj
        for obj in objects_after_first_irs
        if obj["pickupable"]
        and obj["objectType"] in VALID_TARGET_TYPES
        and obj["objectType"] not in object_types_to_not_move
    ]

    valid_pickupable_ids_after_first_irs_to_room = env.obj_id_with_cond_to_room(
        source=valid_pickupables_after_first_irs
    )

    valid_pickupables_in_room_after_first_irs = []
    other_pickupables = [
        obj
        for obj in objects_after_first_irs
        if obj["pickupable"]
        and (
            obj["objectType"] not in VALID_TARGET_TYPES
            or obj["objectType"] in object_types_to_not_move
        )
    ]

    for pobj in valid_pickupables_after_first_irs:
        if (
            valid_pickupable_ids_after_first_irs_to_room[pobj["objectId"]]
            != single_room
        ):
            other_pickupables.append(pobj)
        else:
            valid_pickupables_in_room_after_first_irs.append(pobj)

    if len(valid_pickupables_in_room_after_first_irs) < num_objs_to_move:
        get_logger().info(
            f"Only {len(valid_pickupables_in_room_after_first_irs)} (< {num_objs_to_move})"
            f" valid pickupables in {single_room} after first irs"
        )
        return None, None

    in_room_set = set(o["objectId"] for o in valid_pickupables_in_room_after_first_irs)
    others_set = set(o["objectId"] for o in other_pickupables)
    all_set = set(o["objectId"] for o in objects_after_first_irs if o["pickupable"])
    if in_room_set | others_set != all_set:
        print("ERROR while grouping pickupables - missing coverage")
        return None, None

    if in_room_set & others_set != set():
        print(
            "ERROR while grouping pickupables - overlapping sets (should be disjoint)"
        )
        return None, None

    return valid_pickupables_in_room_after_first_irs, other_pickupables


def generate_one_rearrangement_given_initial_conditions(
    env: RearrangeProcTHOREnvironment,
    scene: str,
    start_kwargs: dict,
    target_kwargs: dict,
    num_objs_to_move: int,
    num_objs_to_open: int,
    single_room: str,
    object_types_to_not_move: Set[str],
    agent_pos: Dict[str, float],
    agent_rot: Dict[str, float],
    allow_putting_objects_away: bool = False,
    possible_openable_ids: Optional[List[str]] = None,
):
    assert possible_openable_ids is not None

    controller = env.controller
    env.procthor_reset(scene_name=scene, force_reset=True)
    controller.step(
        "TeleportFull", horizon=0, standing=True, rotation=agent_rot, **agent_pos,
    )
    if not controller.last_event.metadata["lastActionSuccess"]:
        print(controller.last_event.metadata["errorMessage"])
        return None, None, None

    if not remove_objects_until_all_have_identical_meshes(controller):
        print("Error initially removing objects")
        return None, None, None

    # Works around a labeling issue in THOR for floor lamps (pickupable)
    excluded_object_ids_for_floor_lamp = [
        o["objectId"]
        for o in controller.last_event.metadata["objects"]
        if o["objectType"] == "FloorLamp"
    ]

    controller.step(
        "InitialRandomSpawn",
        **start_kwargs,
        excludedObjectIds=excluded_object_ids_for_floor_lamp,
    )
    if not controller.last_event.metadata["lastActionSuccess"]:
        print(controller.last_event.metadata["errorMessage"])
        return None, None, None

    for _ in range(12):
        # This shouldn't be necessary but we run these actions
        # to let physics settle.
        controller.step("Pass")

    # get initial and post random spawn object data
    objects_after_first_irs = copy.deepcopy(env.objects())

    if any(o["isBroken"] for o in objects_after_first_irs):
        print("Broken objects after first irs")
        return None, None, None

    start_openness = {}
    object_ids_to_open = None
    if num_objs_to_open > 0:
        object_ids_to_open, start_openness = try_to_open(
            env,
            single_room,
            objects_after_first_irs,
            num_objs_to_open,
            possible_openable_ids,
        )

        if object_ids_to_open is None:
            return None, None, None

        # What if opening moved a pickupable? Get the updated list!
        objects_after_first_irs = copy.deepcopy(env.objects())

        if any(o["isBroken"] for o in objects_after_first_irs):
            print("Broken objects after opening")
            return None, None, None

    valid_pickupables_in_room_after_first_irs, other_pickupables = group_pickupables(
        env,
        objects_after_first_irs,
        object_types_to_not_move,
        single_room,
        num_objs_to_move,
    )
    if valid_pickupables_in_room_after_first_irs is None:
        return None, None, None

    # now, choose which objects to move

    random.shuffle(valid_pickupables_in_room_after_first_irs)
    if num_objs_to_move == 0:
        unmoved_obj_ids = [
            o["objectId"] for o in valid_pickupables_in_room_after_first_irs
        ]
        moved_obj_ids = []
    else:
        unmoved_obj_ids = [
            o["objectId"]
            for o in valid_pickupables_in_room_after_first_irs[:-num_objs_to_move]
        ]
        moved_obj_ids = [
            o["objectId"]
            for o in valid_pickupables_in_room_after_first_irs[-num_objs_to_move:]
        ]

    if allow_putting_objects_away:
        # If we're having a really hard time shuffling objects successfully, then let's
        # move some of the objects we don't care about (i.e. the ones whose position won't change)
        # into cupboards/drawers/etc so that there is more space.

        hide_kwargs = dict(start_kwargs)
        hide_kwargs["forceVisible"] = False

        controller.step(
            "InitialRandomSpawn",
            **hide_kwargs,
            excludedObjectIds=moved_obj_ids + excluded_object_ids_for_floor_lamp,
        )
        if not controller.last_event.metadata["lastActionSuccess"]:
            print(controller.last_event.metadata["errorMessage"])
            return None, None, None

        objects_after_first_irs = copy.deepcopy(env.objects())
        if any(o["isBroken"] for o in objects_after_first_irs):
            print("Broken objects after putting objects away")
            return None, None, None

        (
            valid_pickupables_in_room_after_first_irs,
            other_pickupables,
        ) = group_pickupables(
            env,
            objects_after_first_irs,
            object_types_to_not_move,
            single_room,
            num_objs_to_move,
        )
        if valid_pickupables_in_room_after_first_irs is None:
            return None, None, None

        # Check all moved_obj_ids are in valid_pickupables_in_room_after_first_irs
        valid_ids_in_room = set(
            o["objectId"] for o in valid_pickupables_in_room_after_first_irs
        )
        if any(oid not in valid_ids_in_room for oid in moved_obj_ids):
            print(
                "ERROR: Some moved_obj_ids missing from target room after putting objects away"
            )
            return None, None, None

    poses_after_first_irs = [
        extract_obj_data(obj)
        for obj in objects_after_first_irs
        if obj["objectId"] in moved_obj_ids
        or (object_ids_to_open is not None and obj["objectId"] in object_ids_to_open)
    ]

    # Ensure all rooms are reachable after first irs from the starting positions
    controller.step(
        "TeleportFull", horizon=0, standing=True, rotation=agent_rot, **agent_pos,
    )
    if not controller.last_event.metadata["lastActionSuccess"]:
        print(controller.last_event.metadata["errorMessage"])
        return None, None, None

    room_reachable, reachability_meta = env.all_rooms_reachable()
    if not room_reachable:
        print(f"Unreachable rooms in {scene} after first irs: {reachability_meta}")
        return None, None, None

    # Now, try to shuffle
    second_stage_success = False
    target_openness: Optional[Dict[str, float]] = None

    object_ids_not_to_move = [
        o["objectId"]
        for o in objects_after_first_irs
        if o["objectId"] in unmoved_obj_ids
    ]

    excluded_ids = list(
        set(object_ids_not_to_move + [o["objectId"] for o in other_pickupables])
    )

    valid_pickupables_in_room_after_shuffle = None
    receptacle_object_ids_in_room = env.static_receptacles_in_room(room_id=single_room)

    for retry_ind in range(20):
        controller.step(
            "InitialRandomSpawn",
            excludedObjectIds=excluded_ids,
            receptacleObjectIds=receptacle_object_ids_in_room,
            **{**target_kwargs, "randomSeed": target_kwargs["randomSeed"] + retry_ind,},
        )

        if not controller.last_event.metadata["lastActionSuccess"]:
            print(
                single_room,
                retry_ind,
                "single_room_shuffling failed InitialRandomSpawn",
            )
            continue

        for _ in range(12):
            # This shouldn't be necessary but we run these actions
            # to let physics settle.
            controller.step("Pass")

        # change the openness of one the same non-pickupable objects
        if num_objs_to_open > 0:
            try:
                target_openness = open_objs(
                    object_ids_to_open=object_ids_to_open, controller=controller
                )
            except (StopIteration, RuntimeError):
                print("Failed opening after shuffling")
                return None, None, None
            except:
                print("Failed opening after shuffling")
                return None, None, None

        objects_after_shuffle = copy.deepcopy(env.objects())
        for o in objects_after_shuffle:
            if o["isBroken"]:
                print(
                    f"In scene {controller.last_event.metadata['sceneName']},"
                    f" object {o['objectId']} broke during setup."
                )
                return None, None, None

        (
            valid_pickupables_in_room_after_shuffle,
            other_pickupables_after_shuffle,
        ) = group_pickupables(
            env,
            objects_after_shuffle,
            object_types_to_not_move,
            single_room,
            num_objs_to_move,
        )

        # Let's check if we didn't fail
        same_room_success = True
        valid_ids_in_room = set(
            o["objectId"] for o in valid_pickupables_in_room_after_shuffle
        )
        for valid in valid_pickupables_in_room_after_first_irs:
            if valid["objectId"] not in valid_ids_in_room:
                same_room_success = False
                break
            valid_ids_in_room.remove(valid["objectId"])
        if not same_room_success:
            continue

        other_ids = set(o["objectId"] for o in other_pickupables_after_shuffle)
        for other in other_pickupables:
            if other["objectId"] not in other_ids:
                same_room_success = False
                break
            other_ids.remove(other["objectId"])
        if not same_room_success:
            continue

        if len(valid_ids_in_room) != 0:
            continue

        all_teleport_success = True
        for o in valid_pickupables_in_room_after_shuffle:
            if o["objectId"] in moved_obj_ids:
                pos = o["position"]
                positions = [
                    {
                        "x": pos["x"] + 0.001 * xoff,
                        "y": pos["y"] + 0.001 * yoff,
                        "z": pos["z"] + 0.001 * zoff,
                    }
                    for xoff in [0, -1, 1]
                    for zoff in [0, -1, 1]
                    for yoff in [0, 1, 2]
                ]
                controller.step(
                    "TeleportObject",
                    objectId=o["objectId"],
                    positions=positions,
                    rotation=o["rotation"],
                    makeUnbreakable=True,
                )
                if not controller.last_event.metadata["lastActionSuccess"]:
                    all_teleport_success = False
                    break
        if all_teleport_success:
            poses_after_shuffle = [
                extract_obj_data(obj)
                for obj in objects_after_shuffle
                if obj["objectId"] in moved_obj_ids
                or (
                    object_ids_to_open is not None
                    and obj["objectId"] in object_ids_to_open
                )
            ]

            all_different = True
            for before, after in zip(
                sorted(poses_after_first_irs, key=lambda x: x["objectId"]),
                sorted(poses_after_shuffle, key=lambda x: x["objectId"]),
            ):
                if before["objectId"] != after["objectId"]:
                    print(
                        "Wrong ids for poses before/after shuffle. This should be impossible"
                    )
                    return None, None, None
                if env.are_poses_equal(before, after):
                    all_different = False
                    break

            if not all_different:
                continue

            second_stage_success = True
            break

    if not second_stage_success:
        print("Failed shuffling")
        return None, None, None

    # Ensure all rooms are reachable after shuffle from the starting positions
    controller.step(
        "TeleportFull", horizon=0, standing=True, rotation=agent_rot, **agent_pos,
    )
    if not controller.last_event.metadata["lastActionSuccess"]:
        print(controller.last_event.metadata["errorMessage"])
        return None, None, None

    room_reachable, reachability_meta = env.all_rooms_reachable()
    if not room_reachable:
        print(f"Unreachable rooms in {scene} after shuffling: {reachability_meta}")
        return None, None, None

    # add moveables and not valid pickupable types
    pickupables_after_first_irs = (
        valid_pickupables_in_room_after_first_irs + other_pickupables
    )
    pickupables_after_shuffle = (
        valid_pickupables_in_room_after_shuffle + other_pickupables
    )

    pickupables_after_first_irs.sort(key=lambda x: x["objectId"])
    pickupables_after_shuffle.sort(key=lambda x: x["objectId"])

    if any(
        o0["objectId"] != o1["objectId"]
        for o0, o1 in zip(pickupables_after_first_irs, pickupables_after_shuffle)
    ):
        print("Pickupable object ids don't match after shuffle!")
        return None, None, None

    # [opened, starting, target]
    return (
        [
            {
                "objectId": open_obj_name,
                "start_openness": start_openness[open_obj_name],
                "target_openness": target_openness[open_obj_name],
            }
            for open_obj_name in start_openness
        ],
        [
            {
                "name": pickupables_after_first_irs[i]["name"],
                "objectName": pickupables_after_first_irs[i]["name"],
                "position": pickupables_after_first_irs[i]["position"],
                "rotation": pickupables_after_first_irs[i]["rotation"],
            }
            for i in range(len(pickupables_after_first_irs))
        ],
        [
            {
                "name": pickupables_after_shuffle[i]["name"],
                "objectName": pickupables_after_shuffle[i]["name"],
                "position": pickupables_after_shuffle[i]["position"],
                "rotation": pickupables_after_shuffle[i]["rotation"],
            }
            for i in range(len(pickupables_after_shuffle))
        ],
    )


def generate_rearrangement_for_scene(
    stage_seed: int,
    scene: str,
    env: RearrangeProcTHOREnvironment,
    scene_reuse_count: int,
    object_types_to_not_move: Set[str],
    max_obj_rearrangements_per_scene: int = 5,
    obj_name_to_avoid_positions: Optional[Dict[str, np.ndarray]] = None,
    force_visible: bool = True,
    place_stationary: bool = True,
    rotation_increment: int = 90,
    reuse_i: int = 0,
    stage: str = "train",
    allow_moveable_in_goal_randomization: bool = False,
    limits: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    if 360 % rotation_increment != 0:
        raise ValueError("Rotation increment must be a factor of 360")

    if obj_name_to_avoid_positions is None:
        obj_name_to_avoid_positions = defaultdict(
            lambda: np.array([[-1000, -1000, -1000]])
        )

    controller = env.controller

    print(f"Scene {stage} {scene} {reuse_i}")

    seed = md5_hash_str_as_int(f"{stage_seed}|{scene}")
    random.seed(seed)

    if limits is None:
        print(f"Re-computing limnits for Scene {stage} {scene} {reuse_i}??????")
        if not (0 < env.num_rooms(scene) <= MAX_ROOMS_IN_HOUSE):
            print(f"{scene} has {env.num_rooms(scene)} rooms. Skipping.")
            return None

        # House loaded in get_scene_limits
        limits = get_scene_limits(env, scene, object_types_to_not_move)
        if limits is None:
            print(f"Cannot use scene {scene}.")
            return None
    else:
        controller = env.controller

        if not env.procthor_reset(scene_name=scene, force_reset=True):
            print(f"Cannot reset scene {scene}")
            return None

        if not remove_objects_until_all_have_identical_meshes(controller):
            print(
                f"Failed to remove_objects_until_all_have_identical_meshes in {scene}"
            )
            return None

    scene_has_openable = 0 != len(
        [
            o
            for o in controller.last_event.metadata["objects"]
            if o["openable"] and not o["pickupable"]
        ]
    )
    if not scene_has_openable:
        warnings.warn(f"HOUSE {scene} HAS NO OPENABLE OBJECTS")

    evt = controller.step("GetReachablePositions")
    rps: List[Dict[str, float]] = evt.metadata["actionReturn"][:]
    rps.sort(key=lambda d: (round(d["x"], 2), round(d["z"], 2)))
    rotations = np.arange(0, 360, rotation_increment)

    room_to_rps = copy.deepcopy(env.room_to_reachable_positions())
    for room, room_rps in room_to_rps.items():
        room_rps.sort(key=lambda d: (round(d["x"], 2), round(d["z"], 2)))

    assert reuse_i < scene_reuse_count

    try_count = 0

    if limits["max_pickupable"] < max_obj_rearrangements_per_scene:
        print(
            f"Limiting max_obj_rearrangements_per_scene"
            f" from {max_obj_rearrangements_per_scene}"
            f" to {limits['max_pickupable']} in {scene}"
        )
        max_obj_rearrangements_per_scene = limits["max_pickupable"]

    # Evenly distribute # of object rearrangements
    if len(limits["room_openables"]) == 0 or all(
        len(x) == 0 for _, x in limits["room_openables"].items()
    ):
        print(
            f"HOUSE {scene} HAS NO OPENABLE OBJECTS FROM LIMITS (scene_has_openable is {scene_has_openable})"
        )
        num_objs_to_open = 0
    else:
        num_objs_to_open = scene_has_openable * (reuse_i % 2)

    num_objs_to_move = (1 - num_objs_to_open) + math.floor(
        max_obj_rearrangements_per_scene * (reuse_i / scene_reuse_count)
    )
    position_count_offset = 0

    if num_objs_to_open > 0:
        possible_rooms = []
        for room_id, x in limits["room_openables"].items():
            possible_rooms.extend([room_id] * len(x))
        st = random.getstate()
        random.seed(1234567)
        random.shuffle(possible_rooms)
        random.setstate(st)
    else:
        possible_rooms = sorted(list(env.room_to_poly.keys()))

    while True:
        try_count += 1
        if try_count > MAX_TRIES + EXTENDED_TRIES:
            print(f"Something wrong with house {scene} reuse_i {reuse_i}. Skipping.")
            return None
        if try_count == MAX_TRIES + 1:
            print(
                f"Something wrong with house {scene} reuse_i {reuse_i}. Trying another room."
            )
            if len(set(possible_rooms)) > 1:
                possible_rooms = [r for r in possible_rooms if r != single_room]
            else:
                return None

        episode_seed_string = f"{scene}|ind_{reuse_i}|tries_{try_count}|counts_{position_count_offset}|seed_{stage_seed}"
        seed = md5_hash_str_as_int(episode_seed_string)
        random.seed(seed)

        # periodically visit all rooms
        single_room = cast(str, possible_rooms[reuse_i % len(possible_rooms)])

        # avoid agent being unable to teleport to position
        # due to object being placed there
        pos = random.choice(room_to_rps[single_room])
        rot = {"x": 0, "y": int(random.choice(rotations)), "z": 0}

        # used to make sure the positions of the objects
        # are not always the same across the same scene.
        start_kwargs = {
            "randomSeed": random.randint(0, int(1e7) - 1),
            "forceVisible": force_visible,
            "placeStationary": place_stationary,
            "excludedReceptacles": ["ToiletPaperHanger"],
            "allowMoveable": allow_moveable_in_goal_randomization,
        }
        target_kwargs = {
            "randomSeed": random.randint(0, int(1e7) - 1),
            "forceVisible": force_visible,
            "placeStationary": place_stationary,
            "excludedReceptacles": ["ToiletPaperHanger"],
            "allowMoveable": allow_moveable_in_goal_randomization,
        }

        (
            opened_data,
            starting_poses,
            target_poses,
        ) = generate_one_rearrangement_given_initial_conditions(
            env=env,
            scene=scene,
            start_kwargs=start_kwargs,
            target_kwargs=target_kwargs,
            num_objs_to_move=num_objs_to_move + position_count_offset,
            num_objs_to_open=num_objs_to_open,
            single_room=single_room,
            object_types_to_not_move=object_types_to_not_move,
            agent_pos=pos,
            agent_rot=rot,
            allow_putting_objects_away=MAX_TRIES >= try_count >= MAX_TRIES // 2,
            possible_openable_ids=limits["room_openables"][single_room]
            if single_room in limits["room_openables"]
            else [],
        )

        if opened_data is None:
            position_count_offset = max(position_count_offset - 1, 0)
            print(f"{episode_seed_string}: Failed during generation.")
            continue

        task_spec_dict = {
            "agent_position": pos,
            "agent_rotation": int(rot["y"]),
            "object_rearrangement_count": int(num_objs_to_move) + int(num_objs_to_open),
            "openable_data": opened_data,
            "starting_poses": starting_poses,
            "target_poses": target_poses,
        }

        try:
            for _ in range(1):
                env.reset(
                    task_spec=RearrangeTaskSpec(scene=scene, **task_spec_dict),
                    raise_on_inconsistency=True,
                )
                assert env.all_rooms_reachable()[0]
                env.shuffle(raise_on_inconsistency=True)
                assert env.all_rooms_reachable()[0]
                assert env.target_room_id is not None
                assert env.current_room == env.target_room_id
        except:
            get_logger().info(
                f"{episode_seed_string}: Inconsistency or room unreachability when reloading task spec."
            )
            continue

        ips, gps, cps = env.poses
        pose_diffs = cast(
            List[Dict[str, Any]], env.compare_poses(goal_pose=gps, cur_pose=cps)
        )
        reachable_positions = env.controller.step("GetReachablePositions").metadata[
            "actionReturn"
        ]

        failed = False
        for gp, cp, pd in zip(gps, cps, pose_diffs):
            if pd["iou"] is not None and pd["iou"] < IOU_THRESHOLD:
                if gp["type"] in object_types_to_not_move:
                    failed = True
                    print(
                        f"{episode_seed_string}: Moved object of excluded type {gp['type']}"
                    )
                    break

            if gp["broken"] or cp["broken"]:
                failed = True
                print(f"{episode_seed_string}: Broken object")
                break

            pose_diff_energy = env.pose_difference_energy(goal_pose=gp, cur_pose=cp)

            if pose_diff_energy != 0:
                obj_name = gp["objectId"]

                # Ensure that objects to rearrange are visible from somewhere
                interactable_poses = env.controller.step(
                    "GetInteractablePoses",
                    objectId=cp["objectId"],
                    positions=reachable_positions,
                ).metadata["actionReturn"]
                if interactable_poses is None or len(interactable_poses) == 0:
                    print(
                        f"{episode_seed_string}: {obj_name} is not visible despite needing to be rearranged."
                    )

                    failed = True
                    break

                if obj_name in obj_name_to_avoid_positions:
                    if cp["pickupable"]:
                        threshold = 0.15
                        start_position = cp["position"]
                        pos_array = np.array(
                            [[start_position[k] for k in ["x", "y", "z"]]]
                        )
                    elif cp["openness"] is not None:
                        threshold = 0.05
                        pos_array = np.array([[cp["openness"]]])
                    else:
                        continue

                    dist = np.sqrt(
                        ((obj_name_to_avoid_positions[obj_name] - pos_array) ** 2).sum(
                            -1
                        )
                    ).min()
                    if dist <= threshold:
                        print(
                            f"{episode_seed_string}: {obj_name} is within the threshold ({dist} <= {threshold})."
                        )
                        failed = True
                        break
        if failed:
            continue

        npos_diff = int(
            sum(
                pd["iou"] is not None and pd["iou"] < IOU_THRESHOLD for pd in pose_diffs
            )
        )
        nopen_diff = int(
            sum(
                pd["openness_diff"] is not None
                and pd["openness_diff"] >= OPENNESS_THRESHOLD
                for pd in pose_diffs
            )
        )

        if npos_diff != num_objs_to_move:
            position_count_offset += (npos_diff < num_objs_to_move) - (
                npos_diff > num_objs_to_move
            )
            position_count_offset = max(position_count_offset, 0)

            print(
                f"{episode_seed_string}: Incorrect amount of objects have moved expected != actual ({num_objs_to_move} != {npos_diff})"
            )
            continue

        if nopen_diff != num_objs_to_open:
            print(
                f"{episode_seed_string}: Incorrect amount of objects have opened expected != actual ({num_objs_to_open} != {nopen_diff})"
            )
            continue

        task_spec_dict["position_diff_count"] = npos_diff
        task_spec_dict["open_diff_count"] = nopen_diff
        task_spec_dict["pose_diff_energy"] = float(
            env.pose_difference_energy(goal_pose=gps, cur_pose=cps).sum()
        )

        if (npos_diff == 0 and nopen_diff == 0) or task_spec_dict[
            "pose_diff_energy"
        ] == 0.0:
            print(
                f"Not enough has moved in {scene}, {pos}, {int(rot['y'])} {start_kwargs}, {target_kwargs}!"
            )
            continue

        if npos_diff > max_obj_rearrangements_per_scene or nopen_diff > 1:
            print(
                f"{episode_seed_string}: Final check failed ({npos_diff} [{max_obj_rearrangements_per_scene} max] pos. diffs,"
                f" {nopen_diff} [1 max] opened)"
            )
            continue

        print(f"{episode_seed_string} SUCCESS")
        return task_spec_dict


def find_limits_for_scene(
    stage_seed: int,
    scene: str,
    env: RearrangeProcTHOREnvironment,
    object_types_to_not_move: Set[str],
    max_obj_rearrangements_per_scene: int = 5,
    scene_reuse_count: int = 0,
    obj_name_to_avoid_positions: Optional[Dict[str, np.ndarray]] = None,
    force_visible: bool = True,
    place_stationary: bool = False,
    rotation_increment: int = 90,
    reuse_i: int = 0,
    stage: str = "train",
    allow_moveable_in_goal_randomization: bool = False,
) -> Optional[Any]:
    if 360 % rotation_increment != 0:
        raise ValueError("Rotation increment must be a factor of 360")

    print(f"Scene {stage} {scene} limits")

    if not (0 < env.num_rooms(scene) <= MAX_ROOMS_IN_HOUSE):
        print(f"{scene} has {env.num_rooms(scene)} rooms. Skipping.")
        return None

    # House loaded in get_scene_limits
    limits = get_scene_limits(env, scene, object_types_to_not_move)
    if limits is None:
        print(f"Cannot use scene {scene}.")
    return limits


def get_scene_to_obj_name_to_seen_positions():
    scene_to_task_spec_dicts = compress_pickle.load(
        os.path.join(STARTER_DATA_DIR, f"train.pkl.gz")
    )

    scene_to_obj_name_to_positions = {}
    for scene in tqdm.tqdm(scene_to_task_spec_dicts):
        obj_name_to_positions = defaultdict(lambda: [])
        for task_spec_dict in scene_to_task_spec_dicts[scene]:
            for od in task_spec_dict["openable_data"]:
                obj_name_to_positions[od["objectId"]].extend(
                    (od["start_openness"], od["target_openness"])
                )

            for sp, tp in zip(
                task_spec_dict["starting_poses"], task_spec_dict["target_poses"]
            ):
                assert sp["name"] == tp["name"]

                position_dist = IThorEnvironment.position_dist(
                    sp["position"], tp["position"]
                )
                rotation_dist = IThorEnvironment.angle_between_rotations(
                    sp["rotation"], tp["rotation"]
                )
                if position_dist >= 1e-2 or rotation_dist >= 5:
                    obj_name_to_positions[sp["name"]].append(
                        [sp["position"][k] for k in ["x", "y", "z"]]
                    )
                    obj_name_to_positions[sp["name"]].append(
                        [tp["position"][k] for k in ["x", "y", "z"]]
                    )
        scene_to_obj_name_to_positions[scene] = {
            k: np.array(v) for k, v in obj_name_to_positions.items()
        }

    return scene_to_obj_name_to_positions


class RearrangeProcTHORDatagenWorker(Worker):
    def create_env(self, **env_args: Any) -> Optional[Any]:
        env = RearrangeProcTHOREnvironment(
            force_cache_reset=True,
            controller_kwargs={
                "scene": "Procedural",
                "x_display": f"0.{self.gpu}" if self.gpu is not None else None,
            },
        )

        return env

    def work(
        self, task_type: Optional[str], task_info: Dict[str, Any]
    ) -> Optional[Any]:
        if task_type == "rearrange":
            (
                scene,
                seed,
                stage,
                reuse_i,
                obj_name_to_avoid_positions,
                limits,
                scene_reuse_count,
            ) = (
                task_info["scene"],
                task_info["seed"],
                task_info["stage"],
                task_info["reuse_i"],
                task_info["obj_name_to_avoid_positions"],
                task_info["limits"],
                task_info["scene_reuse_count"],
            )

            mode, idx = tuple(scene.split("_"))
            self.env._houses.mode(mode)

            data = generate_rearrangement_for_scene(
                stage_seed=seed,
                scene=scene,
                env=self.env,
                scene_reuse_count=scene_reuse_count,
                object_types_to_not_move=OBJECT_TYPES_TO_NOT_MOVE,
                obj_name_to_avoid_positions=obj_name_to_avoid_positions,
                reuse_i=reuse_i,
                stage=stage,
                limits=limits,
            )

            return data  # Warning: it may be None!
        elif task_type == "find_limits":
            (scene, seed, stage, reuse_i, obj_name_to_avoid_positions) = (
                task_info["scene"],
                task_info["seed"],
                task_info["stage"],
                task_info["reuse_i"],
                task_info["obj_name_to_avoid_positions"],
            )

            mode, idx = tuple(scene.split("_"))
            self.env._houses.mode(mode)

            data = find_limits_for_scene(
                stage_seed=seed,
                scene=scene,
                env=self.env,
                object_types_to_not_move=OBJECT_TYPES_TO_NOT_MOVE,
                obj_name_to_avoid_positions=obj_name_to_avoid_positions,
                reuse_i=reuse_i,
                stage=stage,
            )

            return data


def args_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action="store_true", default=False)
    parser.add_argument("--train_unseen", "-t", action="store_true", default=False)
    parser.add_argument("--mode", "-m", default="train")
    return parser.parse_args()


class RearrangeProcTHORDatagenManager(Manager):
    def work(
        self,
        task_type: Optional[str],
        task_info: Dict[str, Any],
        success: bool,
        result: Any,
    ) -> None:
        if task_type is None:
            args = args_parsing()

            stage_seeds = get_random_seeds()

            if args.mode == "train":
                self.scene_reuse_count = 20
            elif args.mode in ["val", "test"]:
                self.scene_reuse_count = 10

            scene_to_obj_name_to_avoid_positions = None
            if args.debug:
                partition = "train" if args.mode == "train" else "valid"
                idxs = "0,1,2"
                stage_to_scenes = {
                    "debug": [f"{partition}_{idx}" for idx in idxs.split(",")]
                }
            elif args.train_unseen:
                stage_to_scenes = {
                    "train_unseen": [
                        f"train_{id}" for id in range(NUM_TRAIN_UNSEEN_EPISODES)
                    ]
                }
                scene_to_obj_name_to_avoid_positions = (
                    get_scene_to_obj_name_to_seen_positions()
                )
            else:
                nums = {
                    "train": NUM_TRAIN_SCENES,
                    "val": NUM_VALID_SCENES,
                    "test": NUM_TEST_SCENES,
                }
                stage_to_scenes = {
                    stage: [f"{stage}_{id}" for id in range(nums[stage])]
                    for stage in [args.mode]  # ("train", "val", "test")
                }

            os.makedirs(STARTER_DATA_DIR, exist_ok=True)

            self.last_save_time = {stage: time.time() for stage in stage_to_scenes}

            self.stage_to_scene_to_rearrangements = {
                stage: {} for stage in stage_to_scenes
            }
            for stage in stage_to_scenes:
                path = os.path.join(STARTER_DATA_DIR, f"{stage}.json")
                if os.path.exists(path):
                    with open(path, "r") as f:
                        self.stage_to_scene_to_rearrangements[stage] = json.load(f)

            for stage in stage_to_scenes:
                for scene in stage_to_scenes[stage]:
                    if scene not in self.stage_to_scene_to_rearrangements[stage]:
                        self.stage_to_scene_to_rearrangements[stage][scene] = [
                            -1
                        ] * self.scene_reuse_count

                    if scene_to_obj_name_to_avoid_positions is not None:
                        obj_name_to_avoid_positions = scene_to_obj_name_to_avoid_positions[
                            scene
                        ]
                    else:
                        obj_name_to_avoid_positions = None

                    self.enqueue(
                        task_type="find_limits",
                        task_info=dict(
                            scene=scene,
                            stage=stage,
                            seed=stage_seeds[stage],
                            reuse_i=-1,
                            obj_name_to_avoid_positions=obj_name_to_avoid_positions,
                        ),
                    )

        elif task_type == "find_limits":
            if result is not None:
                task_info["limits"] = result
                for reuse_i in range(
                    self.scene_reuse_count
                ):  # if not args.debug else 4):
                    if (
                        self.stage_to_scene_to_rearrangements[task_info["stage"]][
                            task_info["scene"]
                        ][reuse_i]
                        == -1
                    ):
                        task_info["reuse_i"] = reuse_i
                        task_info["scene_reuse_count"] = self.scene_reuse_count
                        self.enqueue(
                            task_type="rearrange", task_info=copy.deepcopy(task_info),
                        )

        elif task_type == "rearrange":
            scene, stage, seed, reuse_i = (
                task_info["scene"],
                task_info["stage"],
                task_info["seed"],
                task_info["reuse_i"],
            )

            scene_to_rearrangements = self.stage_to_scene_to_rearrangements[stage]
            scene_to_rearrangements[scene][reuse_i] = result

            num_missing = len([ep for ep in scene_to_rearrangements[scene] if ep == -1])

            if num_missing == 0:
                get_logger().info(self.info_header + f": Completed {stage} {scene}")

            for stage in self.last_save_time:
                if self.all_work_done or (
                    time.time() - self.last_save_time[stage] > 30 * 60
                ):
                    get_logger().info(self.info_header + f": Saving {stage}")

                    with open(
                        os.path.join(STARTER_DATA_DIR, f"{stage}.json"), "w"
                    ) as f:
                        json.dump(self.stage_to_scene_to_rearrangements[stage], f)

                    compress_pickle.dump(
                        obj=self.stage_to_scene_to_rearrangements[stage],
                        path=os.path.join(STARTER_DATA_DIR, f"{stage}.pkl.gz"),
                        pickler_kwargs={
                            "protocol": 4,
                        },  # Backwards compatible with python 3.6
                    )

                    self.last_save_time[stage] = time.time()
        else:
            raise ValueError(f"Unknown task type {task_type}")


if __name__ == "__main__":
    args = args_parsing()

    print(f"Using args {args}")

    assert args.mode in ["val", "test", "train"]
    if args.train_unseen:
        assert args.mode == "train"

    RearrangeProcTHORDatagenManager(
        worker_class=RearrangeProcTHORDatagenWorker,
        env_args={},
        workers=max((3 * mp.cpu_count()) // 4, 1)
        if platform.system() == "Linux" and not args.debug
        else 1,
        ngpus=torch.cuda.device_count(),
        die_on_exception=False,
        verbose=True,
        debugging=args.debug,
        sleep_between_workers=1.0,
    )
