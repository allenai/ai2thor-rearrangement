import random
from collections import defaultdict
from typing import List, Dict, Set, Optional, Any, Hashable, Sequence

from ai2thor.controller import Controller

from datagen.datagen_constants import OBJECT_TYPES_THAT_CAN_HAVE_IDENTICAL_MESHES
from rearrange_constants import OPENNESS_THRESHOLD


def filter_pickupable(
    objects: List[Dict], object_types_to_not_move: Set[str]
) -> List[Dict]:
    """Filters object data only for pickupable objects."""
    return [
        obj
        for obj in objects
        if obj["pickupable"] and not obj["objectType"] in object_types_to_not_move
    ]


def get_random_seeds(max_seed: int = int(1e8)) -> Dict[str, int]:
    # Generate random seeds for each stage

    # Train seed
    random.seed(1329328939)
    train_seed = random.randint(0, max_seed - 1)

    # Train unseen seed
    random.seed(709384928)
    train_unseen_seed = random.randint(0, max_seed - 1)

    # val seed
    random.seed(3348958620)
    val_seed = random.randint(0, max_seed - 1)

    # test seed
    random.seed(289123396)
    test_seed = random.randint(0, max_seed - 1)

    # Debug seed
    random.seed(239084231)
    debug_seed = random.randint(0, max_seed - 1)

    return {
        "train": train_seed,
        "train_unseen": train_unseen_seed,
        "val": val_seed,
        "valid": val_seed,
        "test": test_seed,
        "debug": debug_seed,
    }


def check_object_opens(
    obj: Dict[str, Any], controller: Controller, return_open_closed: bool = False
):
    controller.step(
        "OpenObject", objectId=obj["objectId"], openness=1.0, forceAction=True,
    )
    obj_opened_fully = controller.last_event.metadata["lastActionSuccess"] and (
        next(
            o
            for o in controller.last_event.metadata["objects"]
            if o["objectId"] == obj["objectId"]
        )["openness"]
        > 0.999
    )

    controller.step(
        "CloseObject", objectId=obj["objectId"], forceAction=True,
    )
    obj_closed_fully = controller.last_event.metadata["lastActionSuccess"] and (
        next(
            o
            for o in controller.last_event.metadata["objects"]
            if o["objectId"] == obj["objectId"]
        )["openness"]
        < 1e-3
    )

    if not return_open_closed:
        return obj_opened_fully and obj_closed_fully
    else:
        return obj_opened_fully and obj_closed_fully, obj_opened_fully, obj_closed_fully


def get_object_by_id(id: str, controller: Controller):
    return next(
        o for o in controller.last_event.metadata["objects"] if o["objectId"] == id
    )


def open_objs(
    object_ids_to_open: List[str], controller: Controller
) -> Dict[str, Optional[float]]:
    """Opens up the chosen pickupable objects if they're openable."""
    out: Dict[str, Optional[float]] = defaultdict(lambda: None)
    for obj_name in object_ids_to_open:
        obj = get_object_by_id(obj_name, controller)

        last_openness = obj["openness"]
        new_openness = last_openness
        while abs(last_openness - new_openness) <= OPENNESS_THRESHOLD:
            new_openness = random.random()

        event = controller.step(
            "OpenObject",
            objectId=obj["objectId"],
            openness=new_openness,
            forceAction=True,
        )
        obj_after_open = get_object_by_id(obj_name, controller)

        if abs(obj_after_open["openness"] - new_openness) > 0.001:
            raise RuntimeError(
                f"In scene {event.metadata['sceneName']}, {obj['objectId']} was supposed to open to {new_openness}"
                f" from {last_openness} but instead reached {obj_after_open['openness']}. Last action success was:"
                f" {event.metadata['lastActionSuccess']}"
            )

        out[obj["objectId"]] = obj_after_open["openness"]
    return out


def remove_objects_until_all_have_identical_meshes(controller: Controller):
    obj_type_to_obj_list = defaultdict(lambda: [])
    for obj in controller.last_event.metadata["objects"]:
        obj_type_to_obj_list[obj["objectType"]].append(obj)

    for obj_type in OBJECT_TYPES_THAT_CAN_HAVE_IDENTICAL_MESHES:
        objs_of_type = list(
            sorted(obj_type_to_obj_list[obj_type], key=lambda x: x["name"])
        )
        random.shuffle(objs_of_type)
        objs_to_remove = objs_of_type[:-1]
        for obj_to_remove in objs_to_remove:
            obj_to_remove_name = obj_to_remove["name"]
            obj_id_to_remove = next(
                obj["objectId"]
                for obj in controller.last_event.metadata["objects"]
                if obj["name"] == obj_to_remove_name
            )
            controller.step("RemoveFromScene", objectId=obj_id_to_remove)
            if not controller.last_event.metadata["lastActionSuccess"]:
                return False
    return True


def mapping_counts(din: Dict[Hashable, Sequence[Hashable]]) -> Dict[Hashable, int]:
    """One-to-many input mapping to one-to-length output mapping"""
    return {k: len(v) for k, v in din.items()}
