import math
import random
import traceback
from collections import OrderedDict, defaultdict
from typing import (
    Dict,
    Any,
    Tuple,
    Optional,
    Callable,
    List,
    Sequence,
    Set,
    cast,
)

import ai2thor.controller
import cv2
import networkx as nx
import numpy as np
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_util import (
    round_to_factor,
    include_object_data,
)
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from utils.procthor_utils import Houses
from rearrange.procthor_rearrange.constants import PROCTHOR_COMMIT_ID
from rearrange.utils import ObjectInteractablePostionsCache
from rearrange.environment import (
    RearrangeTaskSpec,
    RearrangeTHOREnvironment,
    RearrangeMode,
)
from rearrange.utils import (
    get_pose_info,
    iou_box_3d,
    PoseMismatchError,
)


class RearrangeProcTHOREnvironment(RearrangeTHOREnvironment):
    """Custom ProcTHOR Controller for the task of object rearrangement.

    # Attributes
    mode : The current mode of rearrangement. Takes one of the values of RearrangeMode
        (RearrangeMode.SNAP or RearrangeMode.MANIPULATE).
    force_cache_reset : Whether or not we should force cache resets when using the `drop_held_object_with_snap` action.
        Setting this value to `False` results in higher FPS at the expense of possibly having `drop_held_object_with_snap`
        work/fail when it shouldn't. Setting `force_cache_reset` to `True` is recommended during validation/testing.
    obj_id_to_walkthrough_start_pose : Dictionary mapping AI2-THOR object ids to their poses (positions & rotations)
        before they were shuffled (i.e. what the agent sees at the start of the walkthrough phase).
         This will be changed after every call to `reset`.
    obj_id_to_unshuffle_start_pose : Same as `obj_id_to_walkthrough_start_pose` but mapping object ids to their poses (positions &
        rotations) just after they were shuffled, i.e. what the agent sees at the start of the unshuffle phase).
    current_task_spec : A `RearrangeTaskSpec` object specifying the current rearrangement task details.
    controller : A ai2thor controller used to execute all the actions.
    shuffle_called : `True` if the objects have been shuffled so that we're in the `unshuffle` phase. Otherwise `False`.
    """

    def __init__(
        self,
        mode: RearrangeMode = RearrangeMode.SNAP,
        force_cache_reset: Optional[bool] = None,
        controller_kwargs: Optional[Dict[str, Any]] = None,
        enhanced_physics_determinism: bool = True,
        thor_commit_id: str = PROCTHOR_COMMIT_ID,
        houses_revision: str = "rearrangement-2022",
        valid_houses_file=None,
    ):
        self._houses = Houses(
            revision=houses_revision, valid_houses_file=valid_houses_file
        )
        self._current_house: Optional[Dict[str, Any]] = None
        self._house_center: Optional[Dict[str, Any]] = None
        self._current_procthor_scene_name: Optional[str] = None
        self._rooms = None
        self._target_room_id = None
        self._cached_shortest_paths = {}
        self._house_graph = None
        self._last_room_id = None
        self._openness_after_reset = {}

        if controller_kwargs is None:
            controller_kwargs = {}

        controller_kwargs["commit_id"] = thor_commit_id
        controller_kwargs.pop("branch", None)

        super().__init__(
            mode, force_cache_reset, controller_kwargs, enhanced_physics_determinism
        )

        # Object poses at start of walkthrough and unshuffle phases.
        # Reset after every call to reset and shuffle respectively.
        self.__dict__.pop("obj_name_to_walkthrough_start_pose")
        self.__dict__.pop("obj_name_to_unshuffle_start_pose")
        self.obj_id_to_walkthrough_start_pose: Optional[Dict[str, Dict]] = None
        self.obj_id_to_unshuffle_start_pose: Optional[Dict[str, Dict]] = None

        # Overwrite cache of where objects can be interacted with
        self._interactable_positions_cache = ObjectInteractablePostionsCache()

    def objects(self):
        with include_object_data(self.controller):
            objs = self.controller.last_event.metadata["objects"]
        return objs

    def drop_held_object_with_snap(self) -> bool:
        """Drop the object in the agent's hand to the target position.

        Exception is raised if shuffle has not yet been called on the current
        episode or the agent is in default mode.

        For this action to work:
            1. The agent must be within 1.5 meters from the goal object's
               position, observed during the walkthrough phase.
            2. The agent must be looking in the direction of where it was
               located in the walkthrough phase.

        Otherwise, the object will be placed in a visible receptacle or
        if this also fails, it will be simply dropped.

        # Returns

        `True` if the drop was successful, otherwise `False`.
        """
        if not self.shuffle_called:
            raise Exception("Must be in unshuffle stage.")
        if not self.mode == RearrangeMode.SNAP:
            raise Exception("Must be in RearrangeMode.SNAP mode.")

        # round positions to 2 decimals
        DEC = 2

        with include_object_data(self.controller):
            event = self.controller.last_event
            held_obj = self.held_object

            if held_obj is None:
                return False

            # When dropping up an object, make it breakable.
            try:
                self.controller.step(
                    "MakeObjectBreakable", objectId=self.held_object["objectId"]
                )
            except:
                if random.random() > 0.8:
                    get_logger().error(
                        "Unable to make object breakable while dropping with snap"
                    )

            agent = event.metadata["agent"]
            goal_pose = self.obj_id_to_walkthrough_start_pose[held_obj["objectId"]]
            goal_pos = goal_pose["position"]
            goal_rot = goal_pose["rotation"]
            good_positions_to_drop_from = self._interactable_positions_cache.get(
                scene_name=self.scene,
                obj={**held_obj, **{"position": goal_pos, "rotation": goal_rot},},
                controller=self.controller if not hasattr(self, "_houses") else self,
                force_cache_refresh=self.force_cache_reset,  # Forcing cache resets when not training.
            )

            def position_to_tuple(position: Dict[str, float]):
                return tuple(round(position[k], DEC) for k in ["x", "y", "z"])

            agent_xyz = position_to_tuple(agent["position"])
            agent_rot = (round(agent["rotation"]["y"] / 90) * 90) % 360
            agent_standing = int(agent["isStanding"])
            agent_horizon = round(agent["cameraHorizon"])

            for valid_agent_pos in good_positions_to_drop_from:
                # Checks if the agent is close enough to the target
                # for the object to be snapped to the target location.
                valid_xyz = position_to_tuple(valid_agent_pos)
                valid_rot = (round(valid_agent_pos["rotation"] / 90) * 90) % 360
                valid_standing = int(valid_agent_pos["standing"])
                valid_horizon = round(valid_agent_pos["horizon"])
                if (
                    valid_xyz == agent_xyz  # Position
                    and valid_rot == agent_rot  # Rotation
                    and valid_standing == agent_standing  # Standing
                    and round(valid_horizon) == agent_horizon  # Horizon
                ):
                    # Try a few locations near the target for robustness' sake
                    positions = [
                        {
                            "x": goal_pos["x"] + 0.001 * xoff,
                            "y": goal_pos["y"] + 0.001 * yoff,
                            "z": goal_pos["z"] + 0.001 * zoff,
                        }
                        for xoff in [0, -1, 1]
                        for zoff in [0, -1, 1]
                        for yoff in [0, 1, 2]
                    ]
                    self.controller.step(
                        action="TeleportObject",
                        objectId=held_obj["objectId"],
                        rotation=goal_rot,
                        positions=positions,
                        forceKinematic=True,
                        allowTeleportOutOfHand=True,
                        makeUnbreakable=True,
                    )
                    break

            if self.held_object is None:
                # If we aren't holding the object anymore, then let's check if it
                # was placed into the right location.
                if self.are_poses_equal(
                    goal_pose=get_pose_info(goal_pose),
                    cur_pose=next(
                        get_pose_info(o)
                        for o in self.last_event.metadata["objects"]
                        if o["objectId"] == goal_pose["objectId"]
                    ),
                    treat_broken_as_unequal=True,
                ):
                    return True
                else:
                    return False

            # We couldn't teleport the object to the target location, let's try placing it
            # in a visible receptacle.
            possible_receptacles = [
                o for o in event.metadata["objects"] if o["visible"] and o["receptacle"]
            ]
            possible_receptacles = sorted(
                possible_receptacles, key=lambda o: (o["distance"], o["objectId"])
            )
            for possible_receptacle in possible_receptacles:
                self.controller.step(
                    action="PutObject",
                    objectId=possible_receptacle["objectId"],
                    **self.physics_step_kwargs,
                )
                if self.controller.last_event.metadata["lastActionSuccess"]:
                    break

            # We failed to place the object into a receptacle, let's just drop it.
            if len(possible_receptacles) == 0 or (
                not self.controller.last_event.metadata["lastActionSuccess"]
            ):
                try:
                    self.controller.step(
                        "DropHeldObjectAhead",
                        forceAction=True,
                        autoSimulation=False,
                        randomMagnitude=0.0,
                        **{**self.physics_step_kwargs, "actionSimulationSeconds": 1.5},
                    )
                except:
                    get_logger().debug(
                        "'DropHeldObjectAhead' failed, using 'DropHandObject'"
                    )
                    self.controller.step(
                        action="DropHandObject", forceAction=True,
                    )

            return False

    @property
    def poses(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return (unshuffle start, walkthrough start, current) pose for every
        object in the scene.

        Can only be called during the unshuffle phase.

        # Returns
        A Tuple of containing three ordered lists of object poses `(unshuffle_start_poses, walkthrough_start_poses, current_poses)`
        such that, for `0 <= i < len(current_poses)`,
        * `unshuffle_start_poses[i]` - corresponds to the pose of the ith object at the start of the unshuffle phase.
        * `walkthrough_start_poses[i]` - corresponds to the pose of the ith object at the start of the walkthrough phase.
        * `current_poses[i]` - corresponds to the pose of the ith object in the current environment.
        During the unshuffle phase is commonly useful to compare `current_poses[i]` against `walkthrough_start_poses[i]`
        to get a sense of the agent's progress towards placing the objects into their correct locations.
        """
        # Ensure we are in the unshuffle phase.
        if not self.shuffle_called:
            raise Exception("shuffle() must be called before accessing poses")

        # Get current object information
        with include_object_data(self.controller):
            obj_id_to_current_obj = self._obj_list_to_obj_id_to_pose_dict(
                self.controller.last_event.metadata["objects"]
            )

        ordered_obj_ids = list(self.obj_id_to_walkthrough_start_pose.keys())

        current_objs_list = []
        for obj_id in ordered_obj_ids:
            if obj_id not in obj_id_to_current_obj:
                # obj_id_to_predicted_obj can have more objects than goal objects
                # (breaking objects can generate new ones)
                # The other way (more goal poses than predicted objs) is a problem, we will
                # assume that the disappeared objects are broken
                if not self._have_warned_about_mismatch:
                    # Don't want to warn many many times during single episode
                    self._have_warned_about_mismatch = True
                    usos = set(self.obj_id_to_unshuffle_start_pose.keys())
                    wsos = set(self.obj_id_to_walkthrough_start_pose.keys())
                    cos = set(obj_id_to_current_obj.keys())
                    get_logger().warning(
                        f"Mismatch between walkthrough start, unshuffle start, and current pose objects."
                        f"\nscene = {self.scene}, index {self.current_task_spec.metrics.get('index')}"
                        f"\nusos-wsos, wsos-usos = {usos - wsos}, {wsos - usos}"
                        f"\ncos-usos, usos-cos = {cos - usos}, {usos - cos}"
                        f"\ncos-wsos, wsos-cos = {cos - wsos}, {wsos - cos}"
                    )
                obj_id_to_current_obj[obj_id] = {
                    **self.obj_id_to_walkthrough_start_pose[obj_id],
                    "isBroken": True,
                    "broken": True,
                    "position": None,
                    "rotation": None,
                    "openness": None,
                }
            current_objs_list.append(obj_id_to_current_obj[obj_id])

        # We build a cache of object poses corresponding to the start of the walkthrough/unshuffle phases
        # as these remain the same until the `reset` function is called.
        if self._sorted_and_extracted_walkthrough_start_poses is None:
            broken_obj_ids = [
                obj_id
                for obj_id in ordered_obj_ids
                if self.obj_id_to_walkthrough_start_pose[obj_id]["isBroken"]
            ]
            if len(broken_obj_ids) != 0:
                if not self.current_task_spec.runtime_sample:
                    # Don't worry about reporting broken objects when using
                    # a "runtime_sample" task spec as these types of things are
                    # more common.
                    get_logger().warning(
                        f"BROKEN GOAL OBJECTS!"
                        f"\nIn scene {self.scene}"
                        f"\ntask spec {self.current_task_spec}"
                        f"\nbroken objects {broken_obj_ids}"
                    )

                # If we find a broken goal object, we will simply pretend as though it was not
                # broken. This means the agent can never succeed in unshuffling, this means it is
                # possible that even a perfect agent will not succeed for some tasks.
                for broken_obj_id in broken_obj_ids:
                    self.obj_id_to_walkthrough_start_pose[broken_obj_id][
                        "isBroken"
                    ] = False
                    self.obj_id_to_unshuffle_start_pose[broken_obj_id][
                        "isBroken"
                    ] = False
                ordered_obj_ids = list(self.obj_id_to_walkthrough_start_pose.keys())

            walkthrough_start_poses = tuple(
                self.obj_id_to_walkthrough_start_pose[k] for k in ordered_obj_ids
            )
            unshuffle_start_poses = tuple(
                self.obj_id_to_unshuffle_start_pose[k] for k in ordered_obj_ids
            )
            self._sorted_and_extracted_unshuffle_start_poses = get_pose_info(
                unshuffle_start_poses
            )
            self._sorted_and_extracted_walkthrough_start_poses = get_pose_info(
                walkthrough_start_poses
            )

        return (
            self._sorted_and_extracted_unshuffle_start_poses,
            self._sorted_and_extracted_walkthrough_start_poses,
            get_pose_info(current_objs_list),
        )

    def create_controller(self):
        """Create the ai2thor controller."""

        assert ("width" in self._controller_kwargs) == (
            "height" in self._controller_kwargs
        ), "Either controller_kwargs must contain either both of width/height or neither."
        self._controller_kwargs["width"] = self._controller_kwargs.get("width", 300)
        self._controller_kwargs["height"] = self._controller_kwargs.get("height", 300)

        assert (
            "scene" in self._controller_kwargs
            and self._controller_kwargs["scene"] == "Procedural"
        )

        self._controller_kwargs["snapToGrid"] = True

        return ai2thor.controller.Controller(**self._controller_kwargs)

    @property
    def scene(self) -> str:
        """Return the current AI2-THOR scene name."""
        return self._current_procthor_scene_name

    def fix_object_names(self, house):
        known_assets = defaultdict(int)
        to_traverse = house["objects"][:]
        while len(to_traverse):
            cur_obj = to_traverse.pop()
            cur_obj["id"] = f'{cur_obj["assetId"]}_{known_assets[cur_obj["assetId"]]}'
            known_assets[cur_obj["assetId"]] += 1
            if "children" in cur_obj:
                to_traverse.extend(cur_obj["children"][:])

        return house

    @staticmethod
    def find_merged_rooms(house: Dict[str, Any]) -> Dict[str, int]:
        to_merge = {}
        highest_room_id = -1
        for wall in house["walls"]:
            room_id = int(wall["roomId"].split("|")[-1])
            if room_id > highest_room_id:
                highest_room_id = room_id
            if "empty" in wall and wall["empty"]:
                wall_extents = "|".join(wall["id"].split("|")[-4:])
                if wall_extents in to_merge:
                    to_merge[wall_extents].append(wall["roomId"])
                else:
                    to_merge[wall_extents] = [wall["roomId"]]

        if len(to_merge) == 0:
            return {}

        room_to_cluster = {}
        next_cluster = highest_room_id + 1
        for wall, rooms in to_merge.items():
            assert len(rooms) == 2
            if rooms[0] not in room_to_cluster and rooms[1] not in room_to_cluster:
                room_to_cluster[rooms[0]] = next_cluster
                room_to_cluster[rooms[1]] = next_cluster
                next_cluster += 1
            elif rooms[0] in room_to_cluster:
                room_to_cluster[rooms[1]] = room_to_cluster[rooms[0]]
            else:
                room_to_cluster[rooms[0]] = room_to_cluster[rooms[1]]

        return room_to_cluster

    def _merge_rooms(self, house: Dict[str, Any]) -> Dict[str, Any]:
        room_to_cluster = self.find_merged_rooms(house)

        cluster_to_rooms = defaultdict(list)
        for room, cluster in room_to_cluster.items():
            cluster_to_rooms[cluster].append(room)
        for cluster in cluster_to_rooms:
            cluster_to_rooms[cluster] = sorted(cluster_to_rooms[cluster])

        # First, fix all doors, walls and windows by replacing merged rooms' ids by the cluster's id:
        for door in house["doors"]:
            for side in ["room0", "room1"]:
                if door[side] in room_to_cluster:
                    door[side + "_backup"] = door[side]
                    door[side] = f"room|{room_to_cluster[door[side]]}"

        for wall in house["walls"]:
            if wall["roomId"] in room_to_cluster:
                wall["roomId_backup"] = wall["roomId"]
                wall["roomId"] = f'room|{room_to_cluster[wall["roomId"]]}'

        for window in house["windows"]:
            for side in ["room0", "room1"]:
                if window[side] in room_to_cluster:
                    window[side + "_backup"] = window[side]
                    window[side] = f"room|{room_to_cluster[window[side]]}"

        # Next, make a new list of rooms
        new_rooms = []
        merged_ids = set()
        ids_to_rooms = {room["id"]: room for room in house["rooms"]}

        for room in house["rooms"]:
            if room["id"] not in room_to_cluster:
                new_rooms.append(room)
                continue
            elif room["id"] in merged_ids:
                continue

            new_room = dict(
                ceilings=[],
                children=[],
                floorMaterial="MergedRoomMaterial",
                floorPolygon=[],
                id=f"room|{room_to_cluster[room['id']]}",
                roomType=f"OpenSpace",
            )

            polygons = []
            for room_id in cluster_to_rooms[room_to_cluster[room["id"]]]:
                merged_ids.add(room_id)
                new_room["children"].append(ids_to_rooms[room_id])

                polygons.append(
                    Polygon(
                        [
                            [corner["x"], corner["z"]]
                            for corner in ids_to_rooms[room_id]["floorPolygon"]
                        ]
                    )
                )

            merged_poly = unary_union(polygons).simplify(tolerance=0.001)

            new_room["floorPolygon"] = [
                dict(x=x, y=0, z=z) for x, z in merged_poly.exterior.coords[:-1]
            ]

            new_rooms.append(new_room)

        house["rooms"] = new_rooms

        return house

    def get_house_by_name(self, scene_name):
        mode, idx = tuple(scene_name.split("_"))

        self._houses.mode(mode if mode in ["train", "test"] else "validation")

        return self.fix_object_names(self._houses[int(idx)])

    def close_doors(self, house):
        for door in house["doors"]:
            if "openness" in door:
                door["openness"] = 0
        return house

    def procthor_reset(
        self,
        scene_name,
        force_reset=True,
        place_agent=True,
        close_doors=False,
        merge_rooms=True,
        house=None,
    ):
        new_scene = scene_name != self.scene

        # If any object broke, we need to reset
        if (
            not new_scene
            and not force_reset
            and any(o["broken"] for o in self.objects() if "broken" in o)
        ):
            get_logger().info(
                f"Forcing reset due to some objects being broken in {self.scene}"
            )
            force_reset = True

        if force_reset or new_scene:
            self.controller.reset()

            # House without merged rooms
            if house is None:
                house = self.get_house_by_name(scene_name)
                if house["proceduralParameters"]["skyboxId"] == "Sky3":
                    house["proceduralParameters"]["skyboxId"] = "Sky2Dusk"
                if close_doors:
                    house = self.close_doors(house)
            else:
                assert close_doors
            self._current_procthor_scene_name = scene_name

            self.controller.step(
                action="CreateHouse", house=house, raise_for_failure=True
            )
        else:
            house = self.current_house

        success = True
        if place_agent:
            self.controller.step(action="TeleportFull", **house["metadata"]["agent"])
            success = self.last_event.metadata["lastActionSuccess"]

        if new_scene:
            if merge_rooms:
                self._current_house = self._merge_rooms(house)
            else:
                self._current_house = house
            self._house_center = None
            self._rooms = None
            self._cached_shortest_paths = {}
            self._house_graph = None

        self._target_room_id = None

        return success

    @property
    def current_house(self):
        return self._current_house

    def _runtime_reset(
        self, task_spec: RearrangeTaskSpec, force_axis_aligned_start: bool
    ):
        """View docstring in base class"""

        raise NotImplementedError

    @staticmethod
    def set_object_poses_params():
        return dict(
            placeStationary=False,
            enablePhysicsJitter=False,
            forceRigidbodySleep=True,
            skipMoveable=True,
        )

    def _task_spec_reset(
        self,
        task_spec: RearrangeTaskSpec,
        force_axis_aligned_start: bool,
        force_reset: bool = True,
        close_doors: bool = False,
        merge_rooms: bool = True,
        house: Optional[Dict[str, Any]] = None,
        raise_on_inconsistency: bool = False,
    ):
        """Initialize a ai2thor environment from a (non-runtime sample) task
        specification (i.e. an exhaustive collection of object poses for the
        walkthrough and unshuffle phase).

        After this call, the environment will be ready for use in the walkthrough phase.

        # Parameters
        task_spec : The RearrangeTaskSpec for this task. `task_spec.runtime_sample` should be `False`.
        force_axis_aligned_start : If `True`, this will force the agent's start rotation to be 'axis aligned', i.e.
            to equal to 0, 90, 180, or 270 degrees.
        """
        assert (
            not task_spec.runtime_sample
        ), "`_task_spec_reset` requires that `task_spec.runtime_sample` is `False`."

        self.current_task_spec = task_spec

        self.procthor_reset(
            self.current_task_spec.scene,
            force_reset=force_reset,
            close_doors=close_doors,
            merge_rooms=merge_rooms,
            house=house,
        )
        if self._enhanced_physics_determinism:
            self.controller.step("PausePhysicsAutoSim")

        obj_ids = [
            obj["objectId"]
            for obj in self.objects()
            if obj["objectType"]
            not in ["Wall", "Floor", "Window", "Doorway", "Doorframe", "Room"]
        ]
        self.controller.step(
            action="SetObjectFilter", objectIds=obj_ids,
        )  # For walkthrough env, we'll further set an empty filter

        if force_axis_aligned_start:
            self.current_task_spec.agent_rotation = round_to_factor(
                self.current_task_spec.agent_rotation, 90
            )

        # set agent position
        pos = self.current_task_spec.agent_position
        rot = {"x": 0, "y": self.current_task_spec.agent_rotation, "z": 0}
        self.controller.step(
            "TeleportFull",
            **pos,
            rotation=rot,
            horizon=0.0,
            standing=True,
            forceAction=True,
        )

        if raise_on_inconsistency:
            ret_val = self.agent_pose_inconsistency(self.current_task_spec)
            if ret_val is not None:
                raise ValueError(ret_val)

        # show object metadata
        with include_object_data(self.controller):
            # arrange walkthrough poses for pickupable objects
            self.controller.step(
                "SetObjectPoses",
                objectPoses=self.current_task_spec.target_poses,
                **self.set_object_poses_params(),
            )
            if not self.controller.last_event.metadata["lastActionSuccess"]:
                get_logger().debug(
                    f"THOR error message: '{self.controller.last_event.metadata['errorMessage']}. Forcing scene reset."
                )
                self._task_spec_reset(
                    task_spec=task_spec,
                    force_axis_aligned_start=force_axis_aligned_start,
                    force_reset=True,
                    close_doors=close_doors,
                    merge_rooms=merge_rooms,
                    house=house,
                    raise_on_inconsistency=raise_on_inconsistency,
                )

            if raise_on_inconsistency:
                ret_val = self.pose_inconsistency(pose_spec_list=task_spec.target_poses)
                if ret_val is not None:
                    raise ValueError(ret_val)

            # open object
            for obj in self.current_task_spec.openable_data:
                self.controller.step(
                    action="OpenObject",
                    objectId=obj["objectId"],
                    openness=obj["target_openness"],
                    forceAction=True,
                    **self.physics_step_kwargs,
                )

            if raise_on_inconsistency and len(self.current_task_spec.openable_data) > 0:
                ret_val = self.openness_inconsistency(
                    task_spec.openable_data[0]["objectId"],
                    task_spec.openable_data[0]["target_openness"],
                )
                if ret_val is not None:
                    raise ValueError(ret_val)

    def reset(
        self,
        task_spec: RearrangeTaskSpec,
        force_axis_aligned_start: bool = False,
        force_reset: bool = True,
        close_doors: bool = False,
        merge_rooms: bool = True,
        house: Optional[Dict[str, Any]] = None,
        raise_on_inconsistency: bool = False,
    ) -> None:
        """Reset the environment with respect to the new task specification.
         The environment will start in the walkthrough phase.
        # Parameters
        task_spec : The `RearrangeTaskSpec` defining environment state.
        force_axis_aligned_start : If `True`, this will force the agent's start rotation to be 'axis aligned', i.e.
            to equal to 0, 90, 180, or 270 degrees.
        """
        if task_spec.runtime_sample:
            self._runtime_reset(
                task_spec=task_spec, force_axis_aligned_start=force_axis_aligned_start
            )
        else:
            self._task_spec_reset(
                task_spec=task_spec,
                force_axis_aligned_start=force_axis_aligned_start,
                force_reset=force_reset,
                close_doors=close_doors,
                merge_rooms=merge_rooms,
                house=house,
                raise_on_inconsistency=raise_on_inconsistency,
            )

        self.shuffle_called = False
        with include_object_data(self.controller):
            self.obj_id_to_walkthrough_start_pose = self._obj_list_to_obj_id_to_pose_dict(
                self.last_event.metadata["objects"]
            )

        self._have_warned_about_mismatch = False
        self._sorted_and_extracted_walkthrough_start_poses = None
        self._sorted_and_extracted_unshuffle_start_poses = None
        self._agent_signals_done = False
        self._interactable_positions_cache.reset_cache()

    def shuffle(
        self, require_reset: bool = False, raise_on_inconsistency: bool = False
    ):
        """Shuffle objects in the environment to start the unshuffle phase."""

        assert not self.shuffle_called

        runtime_sample = self.current_task_spec.runtime_sample
        if runtime_sample:
            self._runtime_shuffle()
        else:
            self._task_spec_shuffle(
                reset=require_reset, raise_on_inconsistency=raise_on_inconsistency
            )

        # Save object metadata
        with include_object_data(self.controller):
            self.obj_id_to_unshuffle_start_pose = self._obj_list_to_obj_id_to_pose_dict(
                self.last_event.metadata["objects"]
            )

            if len(self.obj_id_to_unshuffle_start_pose) != len(
                self.obj_id_to_walkthrough_start_pose
            ):
                if runtime_sample or require_reset:
                    walkthrough_start_obj_ids = set(
                        self.obj_id_to_walkthrough_start_pose.keys()
                    )
                    unshuffle_start_obj_ids = set(
                        self.obj_id_to_unshuffle_start_pose.keys()
                    )
                    raise PoseMismatchError(
                        "Irrecoverable difference between walkthrough and unshuffle phase objects."
                        f"\ng-i, i-g = {walkthrough_start_obj_ids - unshuffle_start_obj_ids},"
                        f" {unshuffle_start_obj_ids - walkthrough_start_obj_ids}"
                    )
                else:
                    self.shuffle(
                        require_reset=True,
                        raise_on_inconsistency=raise_on_inconsistency,
                    )

        self.shuffle_called = True
        self._agent_signals_done = False

    def openness_inconsistency(self, objectId, spec_openness, tol=0.05):
        found = False
        for obj in self.objects():
            if obj["objectId"] == objectId:
                if found:
                    return "Duplicate objectId"
                found = True
                if abs(obj["openness"] - spec_openness) > tol:
                    return "Wrong openness"
        if not found:
            return "Missing objectId"
        return None

    def pose_inconsistency(self, pose_spec_list, open_tol=0.05, min_iou=0.9):
        shuffled_name_to_spec = {o["name"]: o for o in pose_spec_list}
        used_names = set()
        for obj in self.objects():
            if obj["name"] in used_names:
                return "Duplicate name"

            if obj["name"] in shuffled_name_to_spec:
                env_pose = cast(Dict[str, Any], get_pose_info(obj))

                spec_pose = dict(env_pose)
                spec_pose["rotation"] = shuffled_name_to_spec[obj["name"]]["rotation"]
                spec_pose["position"] = shuffled_name_to_spec[obj["name"]]["position"]

                if not self.are_poses_equal(
                    spec_pose,
                    env_pose,
                    min_iou=min_iou,
                    open_tol=open_tol,
                    treat_broken_as_unequal=True,
                ):
                    return "Wrong pose"

                shuffled_name_to_spec.pop(obj["name"])
                used_names.add(obj["name"])

        if len(shuffled_name_to_spec) != 0:
            return "Missing name"
        return None

    def agent_pose_inconsistency(
        self, task_spec, rot_tol=1.0, pos_tol=0.05, hor_tol=1.0
    ):
        agent_meta = self.last_event.metadata["agent"]
        env_pos = [agent_meta["position"][x] for x in "xz"]
        env_rot = agent_meta["rotation"]["y"] % 360
        env_hor = agent_meta["cameraHorizon"]
        env_stand = agent_meta["isStanding"]

        spec_pos = [task_spec.agent_position[x] for x in "xz"]
        spec_rot = task_spec.agent_rotation % 360
        spec_hor = 0.0
        spec_stand = True

        if sum((ep - sp) ** 2 for ep, sp in zip(env_pos, spec_pos)) > pos_tol:
            return "Wrong agent position"

        if abs(env_rot - spec_rot) > rot_tol:
            return "Wrong agent rotation"

        if abs(env_hor - spec_hor) > hor_tol:
            return "Wrong agent horizon"

        if env_stand != spec_stand:
            return "Wrong agent standing"

        return None

    def _task_spec_shuffle(self, reset: bool = False, raise_on_inconsistency=False):
        """Shuffle objects in the environment to start the unshuffle phase
        using the current task specification.

        Also resets the agent's position to its start position.
        """
        assert not (self.current_task_spec.runtime_sample or self.shuffle_called)

        task_spec = self.current_task_spec

        if reset:
            self.procthor_reset(self.scene)

            obj_ids = [
                obj["objectId"]
                for obj in self.objects()
                if obj["objectType"]
                not in ["Wall", "Floor", "Window", "Doorway", "Doorframe", "Room"]
            ]
            self.controller.step(
                action="SetObjectFilter", objectIds=obj_ids,
            )

            if self._enhanced_physics_determinism:
                self.controller.step("PausePhysicsAutoSim")

        # set agent position
        pos = task_spec.agent_position
        rot = {"x": 0, "y": task_spec.agent_rotation, "z": 0}
        self.controller.step(
            "TeleportFull",
            **pos,
            rotation=rot,
            horizon=0.0,
            standing=True,
            forceAction=True,
        )

        if raise_on_inconsistency:
            ret_val = self.agent_pose_inconsistency(task_spec)
            if ret_val is not None:
                raise ValueError(ret_val)

        # open objects
        for obj in task_spec.openable_data:
            self.controller.step(
                action="OpenObject",
                objectId=obj["objectId"],
                openness=obj["start_openness"],
                forceAction=True,
                **(
                    self.physics_step_kwargs
                    if obj is task_spec.openable_data[-1]
                    else {}
                ),
            )

        if raise_on_inconsistency and len(self.current_task_spec.openable_data) > 0:
            ret_val = self.openness_inconsistency(
                task_spec.openable_data[0]["objectId"],
                task_spec.openable_data[0]["start_openness"],
            )
            if ret_val is not None:
                raise ValueError(ret_val)

        # arrange unshuffle start poses for pickupable objects
        self.controller.step(
            "SetObjectPoses",
            objectPoses=task_spec.starting_poses,
            **self.set_object_poses_params(),
        )
        assert self.controller.last_event.metadata[
            "lastActionSuccess"
        ], self.controller.last_event.metadata["errorMessage"]

        if raise_on_inconsistency:
            ret_val = self.pose_inconsistency(pose_spec_list=task_spec.starting_poses)
            if ret_val is not None:
                raise ValueError(ret_val)

    @staticmethod
    def _obj_list_to_obj_id_to_pose_dict(objects: List[Dict[str, Any]]) -> OrderedDict:
        """Helper function to transform a list of object data dicts into a
        dictionary."""
        objects = [
            o
            for o in objects
            if o["openable"] or o.get("objectOrientedBoundingBox") is not None
        ]
        d = OrderedDict(
            (o["objectId"], o) for o in sorted(objects, key=lambda x: x["objectId"])
        )
        assert len(d) == len(objects)
        return d

    @property
    def room_to_poly(self):
        if self._rooms is None:
            room_to_poly = {}
            for room in self.current_house["rooms"]:
                coords = [[corner["x"], corner["z"]] for corner in room["floorPolygon"]]
                room_to_poly[room["id"]] = Polygon(coords)
            self._rooms = room_to_poly
        return self._rooms

    def randomized_reachable_positions(self):
        meta = self.controller.step("GetReachablePositions").metadata
        if meta["lastActionSuccess"]:
            rps = meta["actionReturn"][:]
            st = random.getstate()
            random.shuffle(rps)
            random.setstate(st)
        else:
            rps = []
        return rps

    def reachable_positions(self):
        meta = self.controller.step("GetReachablePositions").metadata
        if meta["lastActionSuccess"]:
            rps = meta["actionReturn"][:]
        else:
            rps = []
        return rps

    def all_rooms_reachable(self, min_points=5):
        rps = self.randomized_reachable_positions()

        nchecks = 0
        rooms_to_check = list(self.room_to_poly.keys())
        counts = {room: 0 for room in rooms_to_check}
        for p in rps:
            point = Point(p["x"], p["z"])
            for it2, room in enumerate(rooms_to_check):
                room_poly = self.room_to_poly[room]
                nchecks += 1
                if room_poly.contains(point):
                    counts[room] += 1
                    if counts[room] == min_points:
                        removed = rooms_to_check.pop(it2)
                        assert removed == room
                        break
            if len(rooms_to_check) == 0:
                return (
                    True,
                    dict(
                        npoints=len(rps),
                        nrooms=len(self.room_to_poly),
                        nunreachable=len(rooms_to_check),
                        nchecks=nchecks,
                        unreachable_rooms=rooms_to_check,
                    ),
                )

        return (
            False,
            dict(
                npoints=len(rps),
                nrooms=len(self.room_to_poly),
                nunreachable=len(rooms_to_check),
                nchecks=nchecks,
                unreachable_rooms=[],
            ),
        )

    @property
    def house_center(self):
        if self._house_center is None:
            x, z = [], []
            for room in self.current_house["rooms"]:
                for point in room["floorPolygon"]:
                    x.append(point["x"])
                    z.append(point["z"])
            self._house_center = dict(x=(min(x) + max(x)) / 2, z=(min(z) + max(z)) / 2)
            wall_height = 3.5
            self._autofov = max(
                np.arctan((self._house_center["z"] - min(z)) / (19.0 - wall_height))
                * 360.0
                / np.pi,
                self.horizontal_to_vertical_fov(
                    np.arctan((self._house_center["x"] - min(x)) / (19.0 - wall_height))
                    * 360.0
                    / np.pi,
                    height=self._controller_kwargs["height"],
                    width=self._controller_kwargs["width"],
                ),
            )
        return self._house_center

    def _get_extrinsics(self):
        # Rotation: hard-coded to 90 degrees around the x-axis
        R = np.zeros((3, 3), dtype=np.float32)
        cax = np.cos(np.pi / 2)
        sax = np.sin(np.pi / 2)
        R[0, 0] = 1
        R[1, 1] = cax
        R[1, 2] = sax
        R[2, 1] = -sax
        R[2, 2] = cax

        R[1, :] = -R[1, :]  # flip y axis

        # Translation: camera placed at scene center with a height of y=19
        t = -R.T @ np.array([self.house_center["x"], 19.0, self.house_center["z"]])

        return np.concatenate((R, t.reshape(3, 1)), axis=1)

    @staticmethod
    def vertical_to_horizontal_fov(
        vertical_fov_in_degrees: float, height: float, width: float
    ):
        assert 0 < vertical_fov_in_degrees < 180
        aspect_ratio = width / height
        vertical_fov_in_rads = (math.pi / 180) * vertical_fov_in_degrees
        return (
            (180 / math.pi)
            * math.atan(math.tan(vertical_fov_in_rads * 0.5) * aspect_ratio)
            * 2
        )

    @staticmethod
    def horizontal_to_vertical_fov(
        horizontal_fov_in_degrees: float, height: float, width: float
    ):
        return RearrangeProcTHOREnvironment.vertical_to_horizontal_fov(
            horizontal_fov_in_degrees, width, height
        )

    @staticmethod
    def _get_intrinsics(h, w, fv_deg):
        fh = (
            RearrangeProcTHOREnvironment.vertical_to_horizontal_fov(fv_deg, h, w)
            * math.pi
            / 180
        )
        fv = fv_deg * math.pi / 180
        k = [
            [w / (2 * np.tan(fh / 2)), 0, w / 2],
            [0, h / (2 * np.tan(fv / 2)), h / 2],
            [0, 0, 1],
        ]
        return np.array(k)

    def _setup_topdown_camera(self):
        if len(self.last_event.third_party_camera_frames) == 0:
            self.controller.step(
                action="AddThirdPartyCamera",
                position=dict(x=self.house_center["x"], y=19, z=self.house_center["z"]),
                rotation=dict(x=90, y=0, z=0),
                fieldOfView=self._autofov,  # 45 makes the largest house of 12.8 x 12.8 sqm fully visible from y=19
            )
            assert self.last_event.metadata["lastActionSuccess"]
            self._camera_intrinsics = self._get_intrinsics(
                self._controller_kwargs["height"],
                self._controller_kwargs["width"],
                fv_deg=self._autofov,  # 45
            )
            self._camera_extrinsics = self._get_extrinsics()

    def topdown_view(self):
        self._setup_topdown_camera()
        return self.last_event.third_party_camera_frames[0]

    def topdown_box(self, obj_dict):
        """Return a set of lines to be e.g. drawn on the topdown view"""
        self._setup_topdown_camera()

        adj = [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ]

        projected_points = []
        for point in obj_dict["axisAlignedBoundingBox"]["cornerPoints"]:
            xw = point + [1.0]
            xc = self._camera_extrinsics @ xw
            xh = self._camera_intrinsics @ xc
            projected_points.append((xh[:2] / xh[2]).tolist())

        lines = []
        for p1, p2 in adj:
            lines.append((projected_points[p1], projected_points[p2]))

        return lines

    @property
    def current_room(self):
        agent_pos = Point(
            [self.last_event.metadata["agent"]["position"][x] for x in "xz"]
        )
        for room, poly in self.room_to_poly.items():
            if poly.contains(agent_pos):
                self._last_room_id = room
                return room
        return None

    def ids_to_objs(self, source=None, id_field: str = "objectId"):
        source = source or self.objects()
        return {obj[id_field]: obj for obj in source}

    def filter_objs(self, cond: Callable[[Dict[str, Any]], bool]):
        return [obj for obj in self.objects() if cond(obj)]

    def spec_obj_id_with_cond_to_room(
        self, cond: Callable[[Dict[str, Any]], bool] = lambda o: True, source=None
    ):
        room_to_poly = self.room_to_poly

        # Ignore any object not explicitly described in the json house definition:
        to_traverse = self.current_house["objects"][:]
        obj_ids = []
        while len(to_traverse):
            obj = to_traverse.pop()
            obj_ids.append(obj["id"])
            if "children" in obj:
                to_traverse.extend(obj["children"][:])

        source = source or self.objects()

        obj_to_room = {}
        for obj in source:
            if obj["objectId"] not in obj_ids or not cond(obj):
                continue

            center = Point([obj["axisAlignedBoundingBox"]["center"][x] for x in "xz"])
            succeeded = False
            for room, poly in room_to_poly.items():
                if poly.contains(center):
                    obj_to_room[obj["objectId"]] = room
                    succeeded = True
                    break

            if not succeeded:
                if obj["objectType"] not in {
                    "Wall",
                    "Window",
                    "Doorframe",
                    "Doorway",
                    "Floor",
                }:
                    get_logger().warning(
                        f"{obj['objectId']}'s center not contained by any room poly"
                    )

        return obj_to_room

    def obj_id_with_cond_to_room(
        self, cond: Callable[[Dict[str, Any]], bool] = lambda o: True, source=None
    ):
        room_to_poly = self.room_to_poly

        source = source or self.objects()

        obj_to_room = {}
        for obj in source:
            if not cond(obj):
                continue

            center = Point([obj["axisAlignedBoundingBox"]["center"][x] for x in "xz"])
            succeeded = False
            for room, poly in room_to_poly.items():
                if poly.contains(center):
                    obj_to_room[obj["objectId"]] = room
                    succeeded = True
                    break

            if not succeeded:
                if obj["objectType"] not in {
                    "Wall",
                    "Window",
                    "Doorframe",
                    "Doorway",
                    "Floor",
                }:
                    get_logger().warning(
                        f"{obj['objectId']}'s center not contained by any room poly"
                    )

        return obj_to_room

    def filtered_poses(
        self, room_id: Optional[str] = None, receps: Optional[Set[str]] = None
    ):
        groups = ["start", "goal", "current"]
        filt = defaultdict(list)
        room_id = room_id or self.current_room
        if room_id is None:
            return None, None, None

        cur_poly = self.room_to_poly[room_id or self.current_room]

        for name, poses in zip(groups, self.poses):
            for pose in poses:
                if cur_poly.contains(Point([pose["position"][x] for x in "xz"])):
                    if receps is not None:
                        if pose["parentReceptacles"] is not None and (
                            any(rec in receps for rec in pose["parentReceptacles"])
                            or any(
                                pose["objectId"].startswith(rec) for rec in receps
                            )  # drawers, etc
                        ):
                            filt[name].append(pose)
                    else:
                        filt[name].append(pose)

        if receps is not None:
            # Ensure all receptacles return the intersection of the objectIds for goal and current
            goal_set = set(o["objectId"] for o in filt["goal"])
            current_set = set(o["objectId"] for o in filt["current"])
            common_set = goal_set & current_set
            for group in ["goal", "current"]:
                filt[group] = [o for o in filt[group] if o["objectId"] in common_set]

        return tuple(filt[group] for group in groups)

    @property
    def house_graph(self):
        if self._house_graph is None:
            graph = nx.Graph()
            for door in self.current_house["doors"]:
                if door["room0"] != door["room1"]:
                    graph.add_edge(door["room0"], door["room1"])
            self._house_graph = graph

        return self._house_graph

    def house_traversal(self, rooms_scanned: Sequence[str] = (), num_attempts=3):
        """Return a list of rooms to be visited, starting with the current one, in order to traverse the house
        with a short number of room changes. Assumes all doors are open."""

        rooms_pending = {
            room_id
            for room_id in self.room_to_poly.keys()
            if room_id not in set(rooms_scanned)
            or not self.all_rearranged_or_broken_in_room(room_id)
        }

        if len(self.current_house["rooms"]) == 1:
            return [self.current_room] if self.current_room in rooms_pending else []

        graph = nx.Graph()
        for door in self.current_house["doors"]:
            if door["room0"] != door["room1"]:
                weight = 1
                if (
                    door["room0"] not in rooms_pending
                    or door["room1"] not in rooms_pending
                ):
                    weight = 2
                graph.add_edge(door["room0"], door["room1"], weight=weight)

        short_path = self.find_short_path(
            graph, source_node=self.current_room, num_attempts=num_attempts
        )

        # Given the approx. shortest path traversing all rooms from the current one,
        # remove all subsequences for rooms not pending
        return self.remove_completed_subpaths(short_path, rooms_pending)

    def shortest_path_to_target_room(self, from_room=None):
        if from_room is None:
            if self.current_room is None:
                return None
            from_room = self.current_room

        if len(self.room_to_poly) < 2:
            return [from_room]

        if from_room not in self._cached_shortest_paths:
            self._cached_shortest_paths[from_room] = nx.shortest_path(
                self.house_graph, source=from_room, target=self.target_room_id
            )

        return self._cached_shortest_paths[from_room]

    @staticmethod
    def find_short_path(
        graph: nx.Graph, source_node: str, num_attempts: int = 3
    ) -> List[str]:
        """Assumes graphs with >= 2 nodes"""

        # greedy traversal - we'll run it N times and pick the shortest path
        method = lambda g, wt: nx.algorithms.approximation.greedy_tsp(
            g, wt, source=source_node
        )

        shortest_path = [""] * graph.number_of_nodes() * graph.number_of_nodes()
        for attempt in range(num_attempts):
            path = nx.algorithms.approximation.traveling_salesman_problem(
                graph, method=method, cycle=True
            )

            new_path = RearrangeProcTHOREnvironment.strip_return_from_cycle(path)

            if len(new_path) < len(shortest_path):
                shortest_path = new_path

            if len(new_path) == graph.number_of_nodes():
                break

        return shortest_path

    @staticmethod
    def strip_return_from_cycle(path):
        seen_rooms, unseen_rooms = set(), set(path)
        for it, room in enumerate(path):
            seen_rooms.add(room)
            if room in unseen_rooms:
                unseen_rooms.remove(room)
                if len(unseen_rooms) == 0:
                    return path[: it + 1]
        raise ValueError("Bug in `strip_return_from_cycle` implementation")

    @staticmethod
    def remove_completed_subpaths(short_path: Sequence[str], rooms_pending: Set[str]):
        """
        Given a short path traversing all rooms from the current one,
        remove all subsequences for rooms not pending
        """

        current_path = short_path[:]
        changes = True

        while changes and len(current_path) > 0:
            new_path = []
            changes = False

            # First, remove non-pending leaf rooms, either at the end of the path or at the end of a loop
            for it, room in enumerate(current_path):
                if (room not in rooms_pending) and (
                    (it == len(current_path) - 1)
                    or (current_path[it - 1] == current_path[it + 1])
                ):
                    changes = True  # signal we changed to iterate again
                else:
                    new_path.append(room)

            if len(new_path) == 0:
                return []

            # Then, merge consecutive identical rooms
            merged_path = [new_path[0]]
            for room in new_path[1:]:
                if room == merged_path[-1]:
                    changes = True  # signal we changed to iterate again
                else:
                    merged_path.append(room)

            current_path = merged_path

        return current_path

    def initial_room_differences(self, room_id=None):
        initial_poses, goal_poses, _ = self.filtered_poses(room_id or self.current_room)
        return self._differences(initial_poses, goal_poses)

    def room_differences(self, room_id=None):
        _, goal_poses, cur_poses = self.filtered_poses(room_id or self.current_room)
        return self._differences(goal_poses, cur_poses)

    def _differences(self, poses1, poses2):
        difs = []
        for gp, cp in zip(poses1, poses2):
            assert gp["objectId"] == cp["objectId"]
            if (
                not gp["broken"] and not cp["broken"]
            ) and not RearrangeProcTHOREnvironment.are_poses_equal(gp, cp):
                difs.append(gp["objectId"])
        return difs

    def house_differences(self):
        _, goal_poses, cur_poses = self.poses
        difs = []
        for gp, cp in zip(goal_poses, cur_poses):
            assert gp["objectId"] == cp["objectId"]
            if (
                not gp["broken"] and not cp["broken"]
            ) and not RearrangeProcTHOREnvironment.are_poses_equal(gp, cp):
                difs.append(gp["objectId"])
        return difs

    def initial_house_differences(self):
        initial_poses, goal_poses, _ = self.poses
        difs = []
        for gp, cp in zip(goal_poses, initial_poses):
            assert gp["objectId"] == cp["objectId"]
            if (
                not gp["broken"] and not cp["broken"]
            ) and not RearrangeProcTHOREnvironment.are_poses_equal(gp, cp):
                difs.append(gp["objectId"])
        return difs

    def room_centroid(self, room_id):
        return np.round(
            np.array(self.room_to_poly[room_id].centroid.coords), decimals=2
        )

    def room_reachable_centroid(self, room_id):
        all_pos = self.randomized_reachable_positions()
        centroid = self.room_centroid(room_id)
        all_pos = np.array(
            [
                [p[x] for x in "xz"]
                for p in all_pos
                if self.room_to_poly[room_id].contains(Point([p[x] for x in "xz"]))
            ]
        )
        if len(all_pos) == 0:
            return None
        all_squared_dists = np.sum((all_pos - centroid.reshape((1, 2))) ** 2, axis=1)
        point = all_pos[np.argmin(all_squared_dists)]
        return point

    def topdown_reachable_locs(self):
        """Return a set of lines to be e.g. drawn on the topdown view"""
        self._setup_topdown_camera()

        projected_points = []
        for point in self.randomized_reachable_positions():
            xw = [point[x] for x in "xyz"] + [1.0]
            xc = self._camera_extrinsics @ xw
            xh = self._camera_intrinsics @ xc
            projected_points.append((xh[:2] / xh[2]).tolist())

        return projected_points

    @staticmethod
    def openable_object_condition(obj: Dict[str, Any]) -> bool:
        return (
            obj["openable"]
            and not obj["pickupable"]
            and obj["objectType"] not in ["Doorway", "Window"]
        )

    def open_in_room(self, roomid: Optional[str]):
        if roomid is None:
            return []
        return sorted(
            [
                obj
                for obj in self.objects()
                if obj["openable"]
                and obj["openness"] > 0.0
                and not obj["pickupable"]
                and obj["objectType"] not in ["Doorway", "Window"]
                and self.room_to_poly[roomid].contains(
                    Point([obj["axisAlignedBoundingBox"]["center"][x] for x in "xz"])
                )
            ],
            key=lambda x: x["objectId"],
        )

    def all_rearranged_or_broken_in_room(self, room_id: str):
        """Return if every object is simultaneously broken or in its correct
        pose.

        The unshuffle agent can make no more progress on its task in the
        case that that every object is either (1) in its correct
        position or (2) broken so that it can never be placed in its
        correct position. This function simply returns whether this is
        the case.
        """
        return all(
            cp["broken"] or self.are_poses_equal(goal_pose=gp, cur_pose=cp)
            for _, gp, cp in zip(*self.filtered_poses(room_id=room_id))
        )

    @staticmethod
    def static_receptacle_cond(obj: Dict[str, Any]) -> bool:
        return (
            obj["receptacle"]
            and not obj["pickupable"]
            and obj["objectType"] not in ["Floor", "Drawer", "ToiletPaperHanger"]
        )

    def static_receptacles_in_room(self, room_id: str) -> List[str]:
        receptacle_ids_to_room = self.obj_id_with_cond_to_room(
            cond=self.static_receptacle_cond
        )

        dout = defaultdict(list)
        for k, v in receptacle_ids_to_room.items():
            dout[v].append(k)

        return sorted(dout[room_id])

    def room_to_static_receptacle_ids(self) -> Dict[str, Set[str]]:
        receptacle_ids_to_room = self.obj_id_with_cond_to_room(
            cond=self.static_receptacle_cond
        )

        dout = defaultdict(set)
        for k, v in receptacle_ids_to_room.items():
            dout[v].add(k)

        return {**dout}

    def object_ids_with_locs(
        self, obj_ids: Sequence[str]
    ) -> List[Tuple[str, Dict[str, float]]]:
        res = []

        remaining_object_ids = set(obj_ids)
        for obj in self.objects():
            if obj["objectId"] in remaining_object_ids:
                pos = obj["axisAlignedBoundingBox"]["center"]
                res.append((obj["objectId"], pos))
                remaining_object_ids.remove(obj["objectId"])
                if len(remaining_object_ids) == 0:
                    break
        for obj_id in remaining_object_ids:
            get_logger().warning(f"Couldn't find {obj_id} in metadata")

        return sorted(res, key=lambda x: x[0])

    def room_graph(
        self, obj_ids_locs: Sequence[Tuple[str, Dict[str, float]]]
    ) -> nx.Graph:
        """
        Returns a fully-connected graph with an `agent` node and `objectId`s nodes
        with edges corresponding to squared Euclidean distances
        """

        agent_loc = self.last_event.metadata["agent"]["position"]
        items = [("agent", cast(Dict[str, float], agent_loc))] + cast(
            List, obj_ids_locs
        )

        # Use Euclidean distance, which is a good approximation inside of a room
        graph = nx.Graph()
        for it, item in enumerate(items):
            for it2 in range(it + 1, len(items)):
                item2 = items[it2]
                dist2 = sum((item[1][x] - item2[1][x]) ** 2 for x in "xz")
                graph.add_edge(item[0], item2[0], weight=dist2)

        return graph

    def make_room_path(
        self,
        graph: Optional[nx.Graph] = None,
        obj_ids_locs: Optional[Sequence[Tuple[str, Dict[str, float]]]] = None,
        obj_ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """
        Short path through static receptacles in room as a TSP with the return part of the cycle stripped.
        If `graph` is provided, `obj_ids_locs` and `obj_ids` are ignored.
        Else, the graph is built with the agent's current location and, if `obj_ids_locs` is provided,
        `obj_ids` is ignored.
        We return the list of object ids in order
        """
        if graph is None:
            if obj_ids_locs is None:
                if obj_ids is None:
                    obj_ids = self.static_receptacles_in_room(self.current_room)
                obj_ids_locs = self.object_ids_with_locs(obj_ids)
            graph = self.room_graph(obj_ids_locs)

        tsp_cycle = self.find_short_path(graph, "agent")
        traversal = self.strip_repeated_nodes(tsp_cycle)
        return traversal[1:]

    @staticmethod
    def strip_repeated_nodes(path):
        new_path = []
        seen_items = set()
        for item in path:
            if item not in seen_items:
                new_path.append(item)
                seen_items.add(item)
        return new_path

    def pose_iou(self, goal_pose, cur_pose):
        position_dist = IThorEnvironment.position_dist(
            goal_pose["position"], cur_pose["position"]
        )
        rotation_dist = IThorEnvironment.angle_between_rotations(
            goal_pose["rotation"], cur_pose["rotation"]
        )
        if position_dist < 1e-2 and rotation_dist < 10.0:
            iou = 1.0
        else:
            try:
                iou = iou_box_3d(goal_pose["bounding_box"], cur_pose["bounding_box"])
            except Exception as _:
                get_logger().warning(
                    "Could not compute IOU, will assume it was 1. Error during IOU computation:"
                    f"\n{traceback.format_exc()}"
                )
                iou = 1.0

        return iou

    def pose_dif_in_recep(self, recep_id):
        """
        Pose differences "observed" in/on the given receptacle,
        which is assumed to be in the current room.
        It ignores broken objects.
        """
        _, goal, current = self.filtered_poses(self.current_room)

        goal_res = []
        current_res = []

        for gp, cp in zip(goal, current):
            if (
                gp["parentReceptacles"] is None
                or recep_id not in gp["parentReceptacles"]
            ) and (
                cp["parentReceptacles"] is None
                or recep_id not in cp["parentReceptacles"]
            ):
                continue
            assert gp["objectId"] == cp["objectId"]
            if (
                not gp["broken"]
                and not cp["broken"]
                and not RearrangeProcTHOREnvironment.are_poses_equal(gp, cp)
            ):
                if (
                    gp["parentReceptacles"] is not None
                    and recep_id in gp["parentReceptacles"]
                ):
                    goal_res.append(gp)
                if (
                    cp["parentReceptacles"] is not None
                    and recep_id in cp["parentReceptacles"]
                ):
                    current_res.append(cp)

        collisions = defaultdict(list)
        for gp in goal:
            if (
                gp["parentReceptacles"] is None
                or recep_id not in gp["parentReceptacles"]
            ):
                continue
            for cp in current:
                if (
                    cp["parentReceptacles"] is None
                    or recep_id not in cp["parentReceptacles"]
                ) or cp["objectId"] == gp["objectId"]:
                    continue

                if self.pose_iou(gp, cp) > 0:
                    collisions[gp["objectId"]].append(cp)

        return goal_res, current_res, {**collisions}

    def topdown_point(self, point):
        if len(point) == 2:
            xw = [point[0], 0.0, point[1], 1.0]
        else:
            xw = [point[x] for x in "xyz"] + [1.0]
        xc = self._camera_extrinsics @ xw
        xh = self._camera_intrinsics @ xc
        return round(float(xh[0] / xh[2])), round(float(xh[1] / xh[2]))

    def plot_floor(self, img, room, color=(0, 255, 0), thickness=2):
        points = []
        for point in room["floorPolygon"]:
            xw = [point[x] for x in "xyz"] + [1.0]
            xc = self._camera_extrinsics @ xw
            xh = self._camera_intrinsics @ xc
            points.append((xh[:2] / xh[2]).tolist())

        for it in range(len(points) - 1):
            pt1 = (round(points[it][0]), round(points[it][1]))
            pt2 = (round(points[it + 1][0]), round(points[it + 1][1]))
            cv2.line(
                img, pt1, pt2, color=color, thickness=thickness,
            )
        pt1 = (round(points[-1][0]), round(points[-1][1]))
        pt2 = (round(points[0][0]), round(points[0][1]))
        cv2.line(
            img, pt1, pt2, color=color, thickness=thickness,
        )

    def show_merged_rooms(self, color=(0, 255, 0), thickness=2, img=None):
        if img is None:
            img = self.topdown_view().copy()

        for room in self.current_house["rooms"]:
            if room["roomType"] != "OpenSpace" or len(room["children"]) == 0:
                continue

            for child in room["children"]:
                self.plot_floor(img, child, color=(255, 0, 0), thickness=1)

            self.plot_floor(img, room, color=color, thickness=thickness)

        return img

    def show_reachable_locs(self, color=(0, 255, 0), thickness=2, radius=2, img=None):
        if img is None:
            img = self.topdown_view().copy()

        points = self.topdown_reachable_locs()
        for point in points:
            pt = (round(point[0]), round(point[1]))
            cv2.circle(img, pt, radius=radius, color=color, thickness=thickness)

        return img

    def show_room_limits(
        self, color=(255, 0, 0), thickness=2, img=None, centroid_radius=2
    ):
        if img is None:
            img = self.topdown_view().copy()

        for room in self.current_house["rooms"]:
            self.plot_floor(img, room, color=color, thickness=thickness)
            if centroid_radius > 0:
                cent = self.room_reachable_centroid(room["id"])
                if cent is not None:
                    cv2.circle(
                        img,
                        self.topdown_point(cent),
                        radius=centroid_radius,
                        color=color,
                        thickness=thickness,
                    )

        return img

    def num_rooms(self, house_name):
        house = self.get_house_by_name(house_name)
        room_to_open_space = self.find_merged_rooms(house)
        num_open_spaces = len(set(list(room_to_open_space.values())))

        # Original number of rooms - the # rooms that will be merged + the # open spaces created
        return len(house["rooms"]) - len(room_to_open_space) + num_open_spaces

    @property
    def target_room_id(self):
        if self._target_room_id is None:
            rooms_with_differences = []
            for room_id in self.room_to_poly:
                room_differences = len(self.initial_room_differences(room_id))
                if room_differences > 0:
                    rooms_with_differences.append((room_differences, room_id))
            assert len(rooms_with_differences) > 0
            if len(rooms_with_differences) > 1:
                get_logger().warning(
                    f"Found differences in more than one room {rooms_with_differences}"
                )
            self._target_room_id = sorted(rooms_with_differences, reverse=True)[0][1]

        return self._target_room_id

    def room_to_reachable_positions(self):
        rps = self.reachable_positions()

        reachables = {room: [] for room in self.room_to_poly}
        for p in rps:
            point = Point(p["x"], p["z"])
            for room, room_poly in self.room_to_poly.items():
                if room_poly.contains(point):
                    reachables[room].append(p)
                    break

        return reachables
