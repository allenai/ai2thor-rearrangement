"""Definitions for a greedy expert for the `Unshuffle` task."""

import copy
import random
from collections import defaultdict
from typing import (
    Dict,
    Tuple,
    Any,
    Optional,
    List,
    TYPE_CHECKING,
    cast,
    Set,
    Union,
)

import networkx as nx
import stringcase
import numpy as np

from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_util import include_object_data

from rearrange.expert import (
    ShortestPathNavigatorTHOR,
    _are_agent_locations_equal,
)
from rearrange.environment import RearrangeMode

from rearrange.procthor_rearrange.environment import RearrangeProcTHOREnvironment

if TYPE_CHECKING:
    from rearrange.procthor_rearrange.tasks import UnshuffleTask

AgentLocKeyType = Tuple[float, float, int, int]  # x, z, rot, hor
AgentFullLocKeyType = Tuple[
    float, float, int, int, float, bool
]  # x, z, rot, hor, y, standing


class ShortestPathNavigatorProcTHOR(ShortestPathNavigatorTHOR):
    """Tracks shortest paths in AI2-THOR environments.

    Assumes 90 degree rotations and fixed step sizes.

    # Attributes
    controller : The AI2-THOR controller in which shortest paths are computed.
    """

    def __init__(
        self,
        env: RearrangeProcTHOREnvironment,
        grid_size: float,
        include_move_left_right: bool = False,
    ):
        """Create a `ShortestPathNavigatorTHOR` instance.

        # Parameters
        env : A RearrangeProcTHOREnvironment which represents the environment in which shortest paths should be
            computed.
        grid_size : The distance traveled by an AI2-THOR agent when taking a single navigational step.
        include_move_left_right : If `True` the navigational actions will include `MoveLeft` and `MoveRight`, otherwise
            they wil not.
        """
        self.env = env
        super().__init__(env.controller, grid_size, include_move_left_right)

    @property
    def scene_name(self) -> str:
        """Current ai2thor scene."""
        return self.env.scene

    @staticmethod
    def get_full_key(
        input_dict: Dict[str, Any], ndigits: int = 2
    ) -> AgentFullLocKeyType:
        """Return a graph node key given an input agent location dictionary."""
        key = ShortestPathNavigatorTHOR.get_key(input_dict=input_dict, ndigits=ndigits)

        assert "standing" in input_dict

        if "y" in input_dict:
            return key + (
                cast(float, input_dict["y"]),
                cast(bool, input_dict["standing"]),
            )
        else:
            return key + (
                cast(float, input_dict["position"]["y"]),
                cast(bool, input_dict["standing"]),
            )

    @staticmethod
    def get_key_from_full(input_key: AgentFullLocKeyType) -> AgentLocKeyType:
        return input_key[:4]

    @staticmethod
    def location_for_full_key(
        key: AgentFullLocKeyType,
    ) -> Dict[str, Union[float, int, bool]]:
        """Return a agent location dictionary given a full key."""
        x, z, rot, hor, y, standing = key
        return dict(x=x, y=y, z=z, rotation=rot, horizon=hor, standing=standing)


class GreedyExploreUnshuffleExpert:
    def __init__(
        self,
        task: "UnshuffleTask",
        shortest_path_navigator: ShortestPathNavigatorProcTHOR,
        max_priority_per_object: int = 3,
        max_priority_per_receptacle: int = 3,
        steps_for_time_pressure: int = 200,
        exploration_enabled: bool = True,
        scan_before_move: bool = True,
    ):
        get_logger().debug(
            f"Expert started for {task.env.scene} (exploration: {exploration_enabled})"
        )
        self.exploration_enabled = exploration_enabled
        self.scan_before_move = scan_before_move

        self.task = task
        assert self.task.num_steps_taken() == 0

        self.shortest_path_navigator = shortest_path_navigator
        self.max_priority_per_object = max_priority_per_object

        self._last_to_target_recep_id: Optional[str] = None
        self.scanned_receps = set()
        self._current_object_target_keys: Optional[Set[AgentLocKeyType]] = None
        self.recep_id_loc_per_room = dict()
        self.cached_locs_for_recep = dict()

        self.max_priority_per_receptacle = max_priority_per_receptacle
        self.recep_id_to_priority: defaultdict = defaultdict(lambda: 0)
        self.visited_recep_ids_per_room = {
            room: set() for room in self.env.room_to_poly
        }
        self.unvisited_recep_ids_per_room = self.env.room_to_static_receptacle_ids()

        self.steps_for_time_pressure = steps_for_time_pressure

        self.last_expert_mode: Optional[str] = None

        self.expert_action_list: List[Optional[int]] = []

        self._last_held_object_id: Optional[str] = None
        self._last_to_interact_object_pose: Optional[Dict[str, Any]] = None
        self._id_of_object_we_wanted_to_pickup: Optional[str] = None
        self.object_id_to_priority: defaultdict = defaultdict(lambda: 0)

        self.shortest_path_navigator.on_reset()
        self.update(action_taken=None, action_success=None)

    @property
    def env(self) -> RearrangeProcTHOREnvironment:
        return self.unshuffle_env

    @property
    def walkthrough_env(self) -> RearrangeProcTHOREnvironment:
        return cast(RearrangeProcTHOREnvironment, self.task.walkthrough_env)

    @property
    def unshuffle_env(self) -> RearrangeProcTHOREnvironment:
        return cast(RearrangeProcTHOREnvironment, self.task.unshuffle_env)

    def _expert_nav_action_to_room(
        self,
        room: str,
        xz_tol: float = 0.75,
        horizon=30,
        future_agent_loc: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Get the shortest path navigational action towards the current room_id's centroid.
        """
        env = self.env
        shortest_path_navigator = self.shortest_path_navigator

        if future_agent_loc is None:
            agent_loc = env.get_agent_location()
        else:
            agent_loc = future_agent_loc
        source_state_key = shortest_path_navigator.get_key(agent_loc)

        goal_xz = env.room_reachable_centroid(room)
        if goal_xz is None:
            get_logger().debug(
                f"ERROR: Unable to find reachable location near {room}'s centroid"
            )
            return None

        goal_loc = dict(x=goal_xz[0], z=goal_xz[1], rotation=0, horizon=horizon,)
        target_key = shortest_path_navigator.get_key(goal_loc)

        action = "Pass"
        if (
            abs(source_state_key[0] - target_key[0]) > xz_tol
            or abs(source_state_key[1] - target_key[1]) > xz_tol
        ):
            try:
                action = shortest_path_navigator.shortest_path_next_action(
                    source_state_key=source_state_key, goal_state_key=target_key,
                )
            except nx.NetworkXNoPath as _:
                action = None

        return action

    def exploration_pose(self, horizon=30):
        if not self.env.last_event.metadata["agent"]["isStanding"]:
            return dict(action="Stand")
        if round(self.env.last_event.metadata["agent"]["cameraHorizon"]) > horizon + 15:
            return dict(action="LookUp")
        if round(self.env.last_event.metadata["agent"]["cameraHorizon"]) < horizon - 15:
            return dict(action="LookDown")
        return None

    def uncovered_ids_on_recep(self, unshuffle_recep_obj, max_objs_to_check=6):
        ids_shuffle = set(unshuffle_recep_obj["receptacleObjectIds"])
        ids_walkthrough = set(
            next(
                r
                for r in self.walkthrough_env.objects()
                if r["objectId"] == unshuffle_recep_obj["objectId"]
            )["receptacleObjectIds"]
        )

        same_set = ids_shuffle & ids_walkthrough
        same_list = list(same_set)

        ids_shuffle = ids_shuffle - same_set
        ids_walkthrough = ids_walkthrough - same_set

        common_fillers = set(
            same_list[: max_objs_to_check - max(len(ids_shuffle), len(ids_walkthrough))]
        )

        return ids_shuffle, ids_walkthrough, common_fillers

    def get_unscanned_receps(self, rooms_to_check, standing=True, horizon=30):
        agent_key = self.shortest_path_navigator.get_key(self.env.get_agent_location())
        all_objects = self.env.objects()

        for room in rooms_to_check:
            recep_ids_to_check = list(self.unvisited_recep_ids_per_room[room])
            for current_recep_id in recep_ids_to_check:
                if current_recep_id in self.scanned_receps:
                    get_logger().debug(
                        f"ERROR: {current_recep_id} already in `self.scanned_receps`."
                    )
                    self.unvisited_recep_ids_per_room[room].remove(current_recep_id)
                    self.visited_recep_ids_per_room[room].add(current_recep_id)
                    continue

                current_recep = next(
                    o for o in all_objects if o["objectId"] == current_recep_id
                )

                (
                    uncovered_cur,
                    uncovered_goal,
                    uncovered_common,
                ) = self.uncovered_ids_on_recep(current_recep)

                if current_recep["objectId"] not in self.cached_locs_for_recep:
                    needs_action = self._expert_nav_action_to_obj(
                        current_recep,
                        force_standing=standing,
                        force_horizon=horizon,
                        objs_on_recep_cur=uncovered_cur.copy(),
                        objs_on_recep_goal=uncovered_goal.copy(),
                        objs_on_recep_both=uncovered_common.copy(),
                    )
                    if needs_action is None:
                        self._expert_nav_action_to_obj(
                            current_recep,
                            objs_on_recep_cur=uncovered_cur,
                            objs_on_recep_goal=uncovered_goal,
                            objs_on_recep_both=uncovered_common,
                        )
                        if len(self._current_object_target_keys):
                            get_logger().debug(
                                f"Access {current_recep_id} by underconstraining the agent pose"
                            )
                    self.cached_locs_for_recep[
                        current_recep["objectId"]
                    ] = self._current_object_target_keys

                if agent_key in self.cached_locs_for_recep[current_recep["objectId"]]:
                    self.visited_recep_ids_per_room[room].add(current_recep_id)
                    self.scanned_receps.add(current_recep_id)
                    self.unvisited_recep_ids_per_room[room].remove(current_recep_id)
                    if current_recep_id == self._last_to_target_recep_id:
                        self._last_to_target_recep_id = None

    def current_direction(self):
        # order is -x, z, x, -z, -x, ...
        # agent_rot in [0, 360):
        agent_rot = self.env.last_event.metadata["agent"]["rotation"]["y"] % 360
        if 225 <= agent_rot < 315:  # 270
            direction = (-1, 0)
        elif 315 <= agent_rot or agent_rot < 45:  # 0 (360)
            direction = (0, 1)
        elif 45 <= agent_rot <= 135:  # 90
            direction = (1, 0)
        else:  # if 135 <= agent_rot < 225:  # 180
            direction = (0, -1)
        return direction

    @staticmethod
    def angle_to_recep(agent_pos, agent_dir, recep_pos):
        agent_to_recep = np.array(recep_pos) - np.array(agent_pos)
        agent_to_recep_dir = agent_to_recep / (np.linalg.norm(agent_to_recep) + 1e-6)
        ang_dist = np.degrees(np.arccos(np.dot(agent_to_recep_dir, agent_dir)))
        # agent follows a clockwise order for scan, so we have to base on RotateRight
        # order is -x, z, x, -z, -x, ...
        if ang_dist > 45:
            if agent_dir[0] == 0:  # z or -z
                if agent_dir[1] == 1:  # z, so next is x
                    if recep_pos[0] < agent_pos[0]:
                        ang_dist = 360 - ang_dist
                else:  # agent_dir[1] == -1:  # -z, so next is -x
                    if recep_pos[0] > agent_pos[0]:
                        ang_dist = 360 - ang_dist
            else:  # agent_dir[1] == 0, so x or -x
                if agent_dir[0] == 1:  # x, so next is -z
                    if recep_pos[1] > agent_pos[1]:
                        ang_dist = 360 - ang_dist
                else:  # agent_dir[0] == -1:  # -x, so next is z
                    if recep_pos[1] < agent_pos[1]:
                        ang_dist = 360 - ang_dist
        return ang_dist

    def prioritize_receps(self):
        agent_loc = self.env.get_agent_location()

        assert self.env.current_room is not None
        recep_ids = self.unvisited_recep_ids_per_room[self.env.current_room]

        rid_to_locs = {
            rid: rloc for rid, rloc in self.recep_id_loc_per_room[self.env.current_room]
        }

        recep_id_locs = [(rid, rid_to_locs[rid]) for rid in recep_ids]

        agent_dir = self.current_direction()
        agent_pos = (agent_loc["x"], agent_loc["z"])

        failed_places_and_min_dist = (float("inf"), float("inf"))
        recep_id_to_go_to = None
        priorities = []
        for recep_id, recep_loc in recep_id_locs:
            if (
                recep_id in self.cached_locs_for_recep
                and len(self.cached_locs_for_recep[recep_id]) == 0
            ):
                continue
            if self.recep_id_to_priority[recep_id] <= self.max_priority_per_receptacle:
                if self.scan_before_move:
                    recep_pos = (recep_loc["x"], recep_loc["z"])
                    ang_dist = self.angle_to_recep(agent_pos, agent_dir, recep_pos)
                    priority_and_dist_to_object = (
                        self.recep_id_to_priority[recep_id],
                        ang_dist,
                        IThorEnvironment.position_dist(
                            agent_loc, recep_loc, ignore_y=True, l1_dist=True
                        ),
                    )
                else:
                    priority_and_dist_to_object = (
                        self.recep_id_to_priority[recep_id],
                        IThorEnvironment.position_dist(
                            agent_loc, recep_loc, ignore_y=True, l1_dist=True
                        ),
                    )

                priorities.append((priority_and_dist_to_object, recep_id))

                if priority_and_dist_to_object < failed_places_and_min_dist:
                    failed_places_and_min_dist = priority_and_dist_to_object
                    recep_id_to_go_to = recep_id

        if len(priorities) > 1:
            priorities = sorted(priorities)
            max_ps = [priorities[0]]
            for it in range(1, len(priorities)):
                # Let's add some hysteresis
                if (
                    priorities[it][0][0] == max_ps[-1][0][0]
                    and priorities[it][0][-1] <= max_ps[-1][0][-1] + 0.75
                ):
                    max_ps.append(priorities[it])
                else:
                    break
            if len(max_ps) > 1:
                for priority, id in max_ps:
                    if id == self._last_to_target_recep_id:
                        return self._last_to_target_recep_id

        return recep_id_to_go_to

    def pose_action_if_compatible(
        self,
        nav_action,
        pose_action,
        horizon,
        current_recep=None,
        standing=None,
        uncovered_cur=None,
        uncovered_goal=None,
        uncovered_common=None,
        room_id=None,
    ):
        """If possible, we prioritize the pose_action for exploration if it isn't None"""

        assert (current_recep is None) != (room_id is None)

        if pose_action is None:
            return nav_action

        future_agent_loc = self.env.get_agent_location()

        if pose_action["action"] == "Stand":
            future_agent_loc["standing"] = True
            incompatible_action = "Crouch"
        elif pose_action["action"] == "LookUp":
            future_agent_loc["horizon"] -= 30
            incompatible_action = "LookDown"
        else:  # if pose_action["action"] == "LookDown":
            future_agent_loc["horizon"] += 30
            incompatible_action = "LookUp"

        if current_recep is not None:
            future_nav_action = self._expert_nav_action_to_obj(
                current_recep,
                force_standing=standing,
                force_horizon=horizon,
                future_agent_loc=future_agent_loc,
                recep_target_keys=self.cached_locs_for_recep[current_recep["objectId"]],
            )
        else:  # if room_id is not None:
            future_nav_action = self._expert_nav_action_to_room(
                cast(str, room_id), horizon=horizon, future_agent_loc=future_agent_loc
            )

        if future_nav_action == incompatible_action:
            return nav_action
        return pose_action

    def scan_room(self, standing=True, horizon=30):
        env = self.env

        if self.env.current_room is None:
            return None

        current_recep_id = self.prioritize_receps()

        self._last_to_target_recep_id = current_recep_id

        if current_recep_id is None:
            # No receptacle left to scan in room. Let lower priority modes take care of the room
            return None

        assert current_recep_id in self.cached_locs_for_recep

        pose_act = self.exploration_pose(horizon=horizon)

        current_recep = next(
            o for o in env.objects() if o["objectId"] == current_recep_id
        )

        nav_needed = self._expert_nav_action_to_obj(
            current_recep,
            force_standing=standing,
            force_horizon=horizon,
            recep_target_keys=self.cached_locs_for_recep[current_recep_id],
        )

        if nav_needed is None:
            if pose_act is not None:
                return pose_act
            get_logger().debug(
                f"Failed to navigate to {current_recep_id} in {env.current_room} during scan."
                f" Increasing place count and (hopefully) attempting a different receptacle."
            )
            self.recep_id_to_priority[current_recep_id] += 1
            return self.scan_room(standing, horizon)

        if nav_needed != "Pass":
            return self.pose_action_if_compatible(
                nav_action=dict(action=nav_needed),
                pose_action=pose_act,
                current_recep=current_recep,
                standing=standing,
                horizon=horizon,
            )

        if len(self._current_object_target_keys) > 0:
            self.scanned_receps.add(current_recep_id)
            self.unvisited_recep_ids_per_room[self.env.current_room].remove(
                current_recep_id
            )
            self.visited_recep_ids_per_room[self.env.current_room].add(current_recep_id)
            return self.rearrange(mode="causal", room_if_fail=True)

        return None

    def scan_house(self, horizon=30):
        env = self.env

        rooms_to_target = env.shortest_path_to_target_room()
        if rooms_to_target is None or len(rooms_to_target) == 1:
            return None

        pose_act = self.exploration_pose(horizon=horizon)

        room_id = rooms_to_target[1]  # skip current room

        if room_id == env.current_room:
            get_logger().debug(
                "ERROR: We're already in the target room while scanning the house."
            )
            return None

        act = self._expert_nav_action_to_room(room_id, horizon=horizon)
        if act == "Pass":
            # We should be in room if we receive Pass, but as per the check above we're not yet there
            get_logger().debug(
                f"ERROR: Received 'Pass' when navigating to {room_id} during scan."
            )
            return None

        if act is None:
            get_logger().debug(f"ERROR: Failed to navigate to {room_id} during scan.")
            return None

        return self.pose_action_if_compatible(
            nav_action=dict(action=act),
            pose_action=pose_act,
            horizon=horizon,
            room_id=room_id,
        )

    @property
    def expert_action(self) -> int:
        """Get the current greedy expert action.

        # Returns An integer specifying the expert action in the current
        state. This corresponds to the order of actions in
        `self.task.action_names()`. For this action to be available the
        `update` function must be called after every step.
        """
        assert self.task.num_steps_taken() == len(self.expert_action_list) - 1
        return self.expert_action_list[-1]

    def update(self, action_taken: Optional[int], action_success: Optional[bool]):
        """Update the expert with the last action taken and whether or not that
        action succeeded."""
        if action_taken is not None:
            assert action_success is not None

            action_names = self.task.action_names()
            last_expert_action = self.expert_action_list[-1]
            agent_took_expert_action = action_taken == last_expert_action
            action_str = action_names[action_taken]

            was_nav_action = any(k in action_str for k in ["move", "rotate", "look"])

            if "pickup_" in action_str and agent_took_expert_action and action_success:
                self._id_of_object_we_wanted_to_pickup = self._last_to_interact_object_pose[
                    "objectId"
                ]

            if "drop_held_object_with_snap" in action_str and agent_took_expert_action:
                if self._id_of_object_we_wanted_to_pickup is not None:
                    self.object_id_to_priority[
                        self._id_of_object_we_wanted_to_pickup
                    ] += 1
                else:
                    self.object_id_to_priority[self._last_held_object_id] += 1

            if "open_by_type" in action_str and agent_took_expert_action:
                self.object_id_to_priority[
                    self._last_to_interact_object_pose["objectId"]
                ] += 1

            if not action_success:
                if was_nav_action:
                    self.shortest_path_navigator.update_graph_with_failed_action(
                        stringcase.pascalcase(action_str)
                    )
                elif (
                    ("pickup_" in action_str or "open_by_type_" in action_str)
                ) and action_taken == last_expert_action:
                    assert self._last_to_interact_object_pose is not None
                    self._invalidate_interactable_loc_for_pose(
                        location=self.task.unshuffle_env.get_agent_location(),
                        obj_pose=self._last_to_interact_object_pose,
                    )
                elif (
                    ("crouch" in action_str or "stand" in action_str)
                    and self.task.unshuffle_env.held_object is not None
                ) and action_taken == last_expert_action:
                    held_object_id = self.task.unshuffle_env.held_object["objectId"]
                    agent_loc = self.task.unshuffle_env.get_agent_location()
                    agent_loc["standing"] = not agent_loc["standing"]
                    self._invalidate_interactable_loc_for_pose(
                        location=agent_loc,
                        obj_pose=self.task.unshuffle_env.obj_id_to_walkthrough_start_pose[
                            held_object_id
                        ],
                    )
            else:
                # If the action succeeded and was not a move action then let's force an update
                # of our currently targeted object
                if not was_nav_action and not (
                    "crouch" in action_str or "stand" in action_str
                ):
                    self._last_to_interact_object_pose = None

        held_object = self.task.unshuffle_env.held_object
        if self.task.unshuffle_env.held_object is not None:
            self._last_held_object_id = held_object["objectId"]

        self._generate_and_record_expert_action()

    @staticmethod
    def _try_to_interact_with_objs_on_recep(
        env, interactable_positions, objs_on_recep, horizon, standing
    ):
        # Try to (greedily) find an interactable position for all/most objects on objs_on_recep
        last_interactable = interactable_positions
        missing_objs_on_recep = copy.copy(objs_on_recep)
        for obj_id in missing_objs_on_recep:
            new_interactable = env._interactable_positions_cache.get(
                scene_name=env.scene,
                obj=next(o for o in env.objects() if o["objectId"] == obj_id),
                controller=env,
                reachable_positions=GreedyExploreUnshuffleExpert.extract_xyz(
                    last_interactable
                ),
                force_horizon=horizon,
                force_standing=standing,
                avoid_teleport=True,
            )
            if len(new_interactable) > 0:
                objs_on_recep.remove(obj_id)
                last_interactable = new_interactable

        return last_interactable

    @staticmethod
    def _try_to_interact_with_objs_on_either_recep(
        env_cur, env_goal, interactable_positions, objs_on_recep, horizon, standing
    ):
        # Try to (greedily) find an interactable position for most objects on objs_on_recep in one of cur/goal setting.
        last_interactable = interactable_positions
        missing_objs_on_recep = copy.copy(objs_on_recep)
        for obj_id in missing_objs_on_recep:
            new_interactable = env_cur._interactable_positions_cache.get(
                scene_name=env_cur.scene,
                obj=next(o for o in env_cur.objects() if o["objectId"] == obj_id),
                controller=env_cur,
                reachable_positions=GreedyExploreUnshuffleExpert.extract_xyz(
                    last_interactable
                ),
                force_horizon=horizon,
                force_standing=standing,
                avoid_teleport=True,
            )
            if len(new_interactable) > 0:
                objs_on_recep.remove(obj_id)
                last_interactable = new_interactable
                continue

            new_interactable = env_goal._interactable_positions_cache.get(
                scene_name=env_goal.scene,
                obj=next(o for o in env_goal.objects() if o["objectId"] == obj_id),
                controller=env_goal,
                reachable_positions=GreedyExploreUnshuffleExpert.extract_xyz(
                    last_interactable
                ),
                force_horizon=horizon,
                force_standing=standing,
                avoid_teleport=True,
            )
            if len(new_interactable) > 0:
                objs_on_recep.remove(obj_id)
                last_interactable = new_interactable

        return last_interactable

    def _search_locs_to_interact_with_objs_on_recep(
        self,
        obj,
        interactable_positions,
        objs_on_recep_cur,
        objs_on_recep_goal,
        objs_on_recep_both,
        force_horizon,
        force_standing,
    ):
        # Try to find an interactable position for all objects on objs_on_recep_cur
        interactable_positions = self._try_to_interact_with_objs_on_recep(
            self.task.unshuffle_env,
            interactable_positions,
            objs_on_recep_cur,  # modified in-place
            force_horizon,
            force_standing,
        )

        # Try to find an interactable position for all objects on objs_on_recep_goal
        interactable_positions = self._try_to_interact_with_objs_on_recep(
            self.task.walkthrough_env,
            interactable_positions,
            objs_on_recep_goal,  # modified in-place
            force_horizon,
            force_standing,
        )

        interactable_positions = self._try_to_interact_with_objs_on_either_recep(
            self.task.unshuffle_env,
            self.task.walkthrough_env,
            interactable_positions,
            objs_on_recep_both,  # modified in-place
            force_horizon,
            force_standing,
        )

        # Try to get close to the target
        obj_loc = tuple(obj["position"][x] for x in "xyz")
        radius = 0.7  # empirically, it seems unlikely to find a valid location closer than 0.7
        new_positions = []
        unused_positions = set(
            tuple(p[x] for x in ["x", "y", "z", "rotation", "standing", "horizon"])
            for p in interactable_positions
        )

        while len(new_positions) == 0:
            available_locs = list(unused_positions)
            for loc in available_locs:
                if sum((loc[x] - obj_loc[x]) ** 2 for x in [0, 2]) <= radius * radius:
                    new_positions.append(loc)
                    unused_positions.remove(loc)
            radius += 0.2

        return [
            {
                x: p[ix]
                for ix, x in enumerate(
                    ["x", "y", "z", "rotation", "standing", "horizon"]
                )
            }
            for p in new_positions
        ]

    @staticmethod
    def crouch_stand_if_needed(interactable_positions, agent_loc, tol=1e-2):
        for gdl in sorted(
            interactable_positions,
            key=lambda ap: ap["standing"] != agent_loc["standing"],
        ):
            if (
                round(
                    abs(agent_loc["x"] - gdl["x"]) + abs(agent_loc["z"] - gdl["z"]), 2
                )
                <= tol
            ):
                if _are_agent_locations_equal(
                    agent_loc, gdl, ignore_standing=True, tol=tol
                ):
                    if agent_loc["standing"] != gdl["standing"]:
                        return "Crouch" if agent_loc["standing"] else "Stand"
                    else:
                        # We are already at an interactable position
                        return "Pass"

        return None

    @staticmethod
    def extract_xyz(full_poses):
        known_xyz = set()
        res_xyz = []
        for pose in full_poses:
            xyz = tuple(pose[x] for x in "xyz")
            if xyz in known_xyz:
                continue
            known_xyz.add(xyz)
            res_xyz.append({x: pose[x] for x in "xyz"})
        return res_xyz

    def _expert_nav_action_to_obj(
        self,
        obj: Dict[str, Any],
        force_standing=None,
        force_horizon=None,
        objs_on_recep_cur: Optional[Set[str]] = None,
        objs_on_recep_goal: Optional[Set[str]] = None,
        objs_on_recep_both: Optional[Set[str]] = None,
        future_agent_loc: Dict[str, Any] = None,
        recep_target_keys: Optional[Set[AgentLocKeyType]] = None,
        use_walkthrough: bool = False,
    ) -> Optional[str]:
        """
        Get the shortest path navigational action towards the object obj.
        The navigational action takes us to a position from which the
        object is interactable.
        """
        env: RearrangeProcTHOREnvironment = self.task.env
        if future_agent_loc is None:
            agent_loc = env.get_agent_location()
        else:
            # We'll assume the reachable positions are the same as for the current agent_loc
            agent_loc = future_agent_loc
        shortest_path_navigator = self.shortest_path_navigator

        interactable_positions = None
        if recep_target_keys is None:
            reachable_positions = env.controller.step("GetReachablePositions").metadata[
                "actionReturn"
            ]

            assert (
                (objs_on_recep_goal is None)
                == (objs_on_recep_cur is None)
                == (objs_on_recep_both is None)
            )

            interactable_positions = env._interactable_positions_cache.get(
                scene_name=env.scene,
                obj=obj,
                controller=env,
                reachable_positions=reachable_positions,
                force_horizon=force_horizon,
                force_standing=force_standing,
                avoid_teleport=objs_on_recep_cur is not None,
            )

            if len(interactable_positions) == 0:
                self._current_object_target_keys = set()
                return None

            if use_walkthrough:
                interactable_positions = self.walkthrough_env._interactable_positions_cache.get(
                    scene_name=env.scene,
                    obj=obj,
                    controller=self.walkthrough_env,
                    reachable_positions=self.extract_xyz(interactable_positions),
                    force_horizon=force_horizon,
                    force_standing=force_standing,
                )

                if len(interactable_positions) == 0:
                    self._current_object_target_keys = set()
                    return None

            if objs_on_recep_cur is not None:
                # We're scanning a receptacle in a room. We'll try
                # to (greedily) find positions where we can interact with all different
                # objects on the receptacle both in walkthrough (goal) and unshuffle (cur) environments
                # For common objects to both envs, we'll try to cover some (best effort with remaining positions)
                interactable_positions = self._search_locs_to_interact_with_objs_on_recep(
                    obj,
                    interactable_positions,
                    objs_on_recep_cur,  # in-place modified
                    objs_on_recep_goal,  # in-place modified
                    objs_on_recep_both,  # in-place modified
                    force_horizon,
                    force_standing,
                )

            full_target_keys = [
                shortest_path_navigator.get_full_key(loc)
                for loc in interactable_positions
            ]
        else:
            full_target_keys = list(recep_target_keys)

        if future_agent_loc is None:
            self._current_object_target_keys = set(full_target_keys)

        if len(full_target_keys) == 0:
            return None

        source_state_key = shortest_path_navigator.get_key(agent_loc)
        target_keys = [
            shortest_path_navigator.get_key_from_full(key) for key in full_target_keys
        ]

        action = "Pass"
        if source_state_key not in target_keys:
            try:
                action = shortest_path_navigator.shortest_path_next_action_multi_target(
                    source_state_key=source_state_key, goal_state_keys=target_keys,
                )
            except nx.NetworkXNoPath as _:
                # Could not find the expert actions
                return None

        if action != "Pass":
            return action
        else:
            tol = 1e-2
            if interactable_positions is None:
                interactable_positions = [
                    shortest_path_navigator.location_for_full_key(key)
                    for key in full_target_keys
                ]
                tol = 2e-2
            return self.crouch_stand_if_needed(
                interactable_positions, agent_loc, tol=tol
            )

    def _invalidate_interactable_loc_for_pose(
        self, location: Dict[str, Any], obj_pose: Dict[str, Any]
    ) -> bool:
        """Invalidate a given location in the `interactable_positions_cache` as
        we tried to interact but couldn't."""
        env = self.unshuffle_env

        interactable_positions = env._interactable_positions_cache.get(
            scene_name=env.scene, obj=obj_pose, controller=env
        )
        for i, loc in enumerate([*interactable_positions]):
            if (
                self.shortest_path_navigator.get_key(loc)
                == self.shortest_path_navigator.get_key(location)
                and loc["standing"] == location["standing"]
            ):
                interactable_positions.pop(i)
                return True
        return False

    def manage_held_object(self) -> Optional[Dict[str, Any]]:
        if self.env.held_object is None:
            return None

        self._last_to_interact_object_pose = None

        # Should navigate to a position where the held object can be placed
        expert_nav_action = self._expert_nav_action_to_obj(
            obj={
                **self.env.held_object,
                **{
                    k: self.env.obj_id_to_walkthrough_start_pose[
                        self.env.held_object["objectId"]
                    ][k]
                    for k in ["position", "rotation"]
                },
            },
        )

        if expert_nav_action is None:
            # Could not find a path to the target, let's just immediately drop the held object
            return dict(action="DropHeldObjectWithSnap")
        elif expert_nav_action == "Pass":
            # We are in a position where we can drop the object, let's do that
            return dict(action="DropHeldObjectWithSnap")
        else:
            return dict(action=expert_nav_action)

    def time_pressure(self):
        if self.task.num_steps_taken() == self.steps_for_time_pressure:
            get_logger().debug(
                f"Expert rushing due to time pressure ({self.steps_for_time_pressure} steps)"
            )
        return self.task.num_steps_taken() >= self.steps_for_time_pressure

    def _log_output_mode(self, expert_mode, action_dict):
        if self.last_expert_mode != expert_mode:
            get_logger().debug(f"{expert_mode} mode")
            self.last_expert_mode = expert_mode
        return action_dict

    def update_visited_receps(self, horizon=30):
        if self.env.current_room is None:
            if self.env._last_room_id is not None:
                rooms_to_check = [self.env._last_room_id]
            else:
                return
        else:
            rooms_to_check = [self.env.current_room]

        for room in rooms_to_check:
            if room not in self.recep_id_loc_per_room:
                recep_ids = self.unvisited_recep_ids_per_room[room]
                self.recep_id_loc_per_room[room] = self.env.object_ids_with_locs(
                    list(recep_ids)
                )

        self.get_unscanned_receps(rooms_to_check, horizon=horizon, standing=True)

    def _generate_expert_action_dict(self, horizon=30) -> Optional[Dict[str, Any]]:
        if self.env.mode != RearrangeMode.SNAP:
            raise NotImplementedError(
                f"Expert only defined for 'easy' mode (current mode: {self.env.mode}"
            )

        try:
            # Try to transport or drop the current object:
            attempt = self.manage_held_object()
            if attempt is not None:
                return self._log_output_mode("held object", attempt)

            if self.exploration_enabled:
                # Check which receptacles are "scanned"/visited from the current location
                # assuming that we cannot scan with a held object (it covers most of the picture)
                self.update_visited_receps(horizon=horizon)

                if not self.time_pressure():
                    # Try to rearrange among seen receptacles (causal rearrangement):
                    attempt = self.rearrange(mode="causal")
                    if attempt is not None:
                        return self._log_output_mode("rearrange (causal)", attempt)

                    # Try to scan the next receptacle:
                    attempt = self.scan_room(horizon=horizon)
                    if attempt is not None:
                        return self._log_output_mode("scan room", attempt)

                    # Try to rearrange the current room (hopefully observed):
                    attempt = self.rearrange(mode="room")
                    if attempt is not None:
                        return self._log_output_mode("rearrange (room)", attempt)

                    # Try to scan/navigate to another room:
                    attempt = self.scan_house(horizon=horizon)
                    if attempt is not None:
                        return self._log_output_mode("scan house", attempt)

            # If I couldn't scan anything, let's use the fallback and try to rearrange the house in the wild
            attempt = self.rearrange(mode="house")
            if attempt is not None:
                return self._log_output_mode("rearrange (house)", attempt)

            if self.exploration_enabled:
                if self.env.target_room_id != self.env.current_room:
                    get_logger().debug(
                        f"WARNING: We cannot generate more actions despite being in {self.env.current_room},"
                        f" away from the target {self.env.target_room_id}. Terminating"
                    )

            return dict(action="Done")

        except:
            import traceback

            get_logger().debug(f"EXCEPTION: Expert failure: {traceback.format_exc()}")
            return None

    def prioritize_objects(self, agent_loc, goal_poses, cur_poses):
        failed_places_and_min_dist = (float("inf"), float("inf"))
        obj_pose_to_go_to = None
        goal_obj_pos = None
        for gp, cp in zip(goal_poses, cur_poses):
            if (
                not gp["broken"]
                and not cp["broken"]
                and self.object_id_to_priority[gp["objectId"]]
                <= self.max_priority_per_object
                and not RearrangeProcTHOREnvironment.are_poses_equal(gp, cp)
            ):
                priority = self.object_id_to_priority[gp["objectId"]]
                priority_and_dist_to_object = (
                    priority,
                    IThorEnvironment.position_dist(
                        agent_loc, gp["position"], ignore_y=True, l1_dist=True
                    ),
                )
                if (
                    self._last_to_interact_object_pose is not None
                    and self._last_to_interact_object_pose["objectId"] == gp["objectId"]
                ):
                    # Set distance to -1 for the currently targeted object
                    priority_and_dist_to_object = (
                        priority_and_dist_to_object[0],
                        -1,
                    )

                if priority_and_dist_to_object < failed_places_and_min_dist:
                    failed_places_and_min_dist = priority_and_dist_to_object
                    obj_pose_to_go_to = cp
                    goal_obj_pos = gp

        return obj_pose_to_go_to, goal_obj_pos

    def rearrange(
        self, mode: str = "causal", room_if_fail=False
    ) -> Optional[Dict[str, Any]]:
        if mode == "causal":
            _, goal_poses, cur_poses = self.env.filtered_poses(
                room_id=self.env.current_room, receps=self.scanned_receps
            )
        elif mode == "room":
            _, goal_poses, cur_poses = self.env.filtered_poses(
                room_id=self.env.current_room
            )
        else:  # if mode == "house"
            _, goal_poses, cur_poses = self.env.poses

        if goal_poses is None or cur_poses is None:
            # This can happen if room_id is None
            return None

        assert len(goal_poses) == len(cur_poses)

        agent_loc = self.env.get_agent_location()

        obj_pose_to_go_to, goal_obj_pos = self.prioritize_objects(
            agent_loc, goal_poses, cur_poses
        )

        self._last_to_interact_object_pose = obj_pose_to_go_to

        if obj_pose_to_go_to is None:
            # Nothing left to rearrange
            if room_if_fail:
                return self.scan_room()
            # If mode is [room, house], we will [scan house, terminate]
            return None

        if (
            obj_pose_to_go_to["openness"] is not None
            and obj_pose_to_go_to["openness"] != goal_obj_pos["openness"]
        ):
            # For open/close, ensure both configurations are interactable
            # (even though we fully close as an intermediate step)
            expert_nav_action = self._expert_nav_action_to_obj(
                obj=obj_pose_to_go_to, use_walkthrough=True
            )
        else:
            expert_nav_action = self._expert_nav_action_to_obj(obj=obj_pose_to_go_to)

        if expert_nav_action is None:
            interactable_positions = self.env._interactable_positions_cache.get(
                scene_name=self.env.scene,
                obj=obj_pose_to_go_to,
                controller=self.env.controller,
            )

            if len(interactable_positions) != 0:
                # Could not find a path to the object, increment the place count of the object and
                # try generating a new action.
                get_logger().debug(
                    f"Could not find a path to {obj_pose_to_go_to['objectId']}"
                    f" in scene {self.task.unshuffle_env.scene}"
                    f" when at position {self.task.unshuffle_env.get_agent_location()}."
                )
            else:
                get_logger().debug(
                    f"Object {obj_pose_to_go_to['objectId']} in scene {self.task.unshuffle_env.scene}"
                    f" has no interactable positions."
                )

            self.object_id_to_priority[obj_pose_to_go_to["objectId"]] += 1

            return self.rearrange(mode=mode)

        if expert_nav_action == "Pass":
            with include_object_data(self.env.controller):
                visible_objects = {
                    o["objectId"]
                    for o in self.env.last_event.metadata["objects"]
                    if o["visible"]
                }

            if obj_pose_to_go_to["objectId"] not in visible_objects:
                if self._invalidate_interactable_loc_for_pose(
                    location=agent_loc, obj_pose=obj_pose_to_go_to
                ):
                    return self.rearrange(mode=mode)

                get_logger().debug(
                    "ERROR: This should not be possible. Failed to invalidate interactable loc for obj pose"
                )
                return None

            # The object of interest is interactable at the moment
            if (
                obj_pose_to_go_to["openness"] is not None
                and obj_pose_to_go_to["openness"] != goal_obj_pos["openness"]
            ):
                return dict(
                    action="OpenByType",
                    objectId=obj_pose_to_go_to["objectId"],
                    openness=goal_obj_pos["openness"],
                )

            if obj_pose_to_go_to["pickupable"]:
                return dict(action="Pickup", objectId=obj_pose_to_go_to["objectId"],)

            # We (likely) have an openable object which has been moved somehow but is not
            # pickupable. We don't know what to do with such an object so we'll set its
            # place count to a large value and try again.
            get_logger().debug(
                f"WARNING: {obj_pose_to_go_to['objectId']} has moved but is not pickupable."
            )
            self.object_id_to_priority[goal_obj_pos["objectId"]] = (
                self.max_priority_per_object + 1
            )

            return self.rearrange(mode=mode)

        # If we are not looking at the object to change, then we should navigate to it
        return dict(action=expert_nav_action)

    def _generate_and_record_expert_action(self):
        """Generate the next greedy expert action and save it to the
        `expert_action_list`."""
        if self.task.num_steps_taken() == len(self.expert_action_list) + 1:
            get_logger().debug(
                f"WARNING: Already generated the expert action at step {self.task.num_steps_taken()}"
            )
            return

        assert self.task.num_steps_taken() == len(
            self.expert_action_list
        ), f"{self.task.num_steps_taken()} != {len(self.expert_action_list)}"
        expert_action_dict = self._generate_expert_action_dict()
        if expert_action_dict is None:
            self.expert_action_list.append(None)
            return

        action_str = stringcase.snakecase(expert_action_dict["action"])
        if action_str not in self.task.action_names():
            current_objectId = expert_action_dict["objectId"]
            current_obj = next(
                o for o in self.task.env.objects() if o["objectId"] == current_objectId
            )
            obj_type = stringcase.snakecase(current_obj["objectType"])
            action_str = f"{action_str}_{obj_type}"
        try:
            self.expert_action_list.append(self.task.action_names().index(action_str))
        except ValueError:
            get_logger().debug(
                f"ERROR: {action_str} is not a valid action for the given task."
            )
            self.expert_action_list.append(None)


if __name__ == "__main__":
    import sys

    from allenact.utils.system import init_logging

    def test(
        stage="train",
        skip_between_tasks=105,
        random_action_prob=0.0,
        explore=True,
        include_other_move_actions=True,
    ):
        import numpy as np

        # noinspection PyUnresolvedReferences
        from baseline_configs.one_phase.procthor.one_phase_rgb_clip_dagger import (
            ProcThorOnePhaseRGBClipResNet50DaggerTrainMultiNodeConfig as Config,
        )

        # noinspection PyUnresolvedReferences
        from rearrange.utils import save_frames_to_mp4

        task_sampler = Config(
            expert_exploration_enabled=explore,
            include_other_move_actions=include_other_move_actions,
        ).make_sampler_fn(
            stage=stage,
            seed=0,
            force_cache_reset=True,
            allowed_scenes=None,  # ["train_0", "train_408"],
            epochs=1,
        )

        k = 0

        # If you want to restart from a particular k:
        task_sampler.reset()
        for _ in range(0):
            k += 1
            next(task_sampler.task_spec_iterator)

        run_tasks = 0
        successful_tasks = 0
        task_lengths = []
        partial_success = []
        num_misplaced = 0
        while task_sampler.length > 0:
            print(k)

            random.seed(k)
            k += 1

            task = task_sampler.next_task()
            if task is None:
                break

            obs = task.get_observations()

            controller = task_sampler.unshuffle_env.controller
            frames = [controller.last_event.frame]
            while not task.is_done():
                if random.random() < random_action_prob or not bool(
                    obs["expert_action"][1]
                ):
                    assert task.action_names()[0] == "done"
                    if random.random() < 0.5:
                        action_to_take = next(
                            it
                            for it, action in enumerate(task.action_names())
                            if "ove" in action and "head" in action
                        )
                    else:
                        action_to_take = random.randint(0, len(task.action_names()) - 1)
                else:
                    action_to_take = obs["expert_action"][0]

                step_result = task.step(action_to_take)
                obs = step_result.observation
                task.unshuffle_env.controller.step("Pass")
                task.walkthrough_env.controller.step("Pass")

                frames.append(controller.last_event.frame)

            run_tasks += 1
            metrics = task.metrics()
            task_lengths.append(len(task.greedy_expert.expert_action_list))
            partial_success.append(
                metrics["unshuffle/prop_fixed"]
                * metrics["unshuffle/num_initially_misplaced"]
            )
            num_misplaced += metrics["unshuffle/num_initially_misplaced"]

            if metrics["unshuffle/prop_fixed"] == 1:
                successful_tasks += 1
                print("Greedy expert success")
            else:
                print("Greedy expert failure", task.metrics()["unshuffle/prop_fixed"])
                _, goal_poses, cur_poses = task.env.poses
                assert len(goal_poses) == len(cur_poses)
                for gp, cp in zip(goal_poses, cur_poses):
                    if (
                        not gp["broken"]
                        and not cp["broken"]
                        and not RearrangeProcTHOREnvironment.are_poses_equal(gp, cp)
                    ):
                        get_logger().info(
                            f"GOAL {gp['objectId']} {gp['position']} {gp['parentReceptacles']}"
                        )
                        get_logger().info(
                            f"RESULT {cp['objectId']} {cp['position']} {cp['parentReceptacles']}"
                        )
                    elif cp["broken"] and not gp["broken"]:
                        get_logger().info(f"broken {gp['type']}")

            print(
                f"Ran tasks {run_tasks} Success rate {successful_tasks / run_tasks * 100:.2f}%"
                f" length {np.mean(task_lengths):.2f} Partial success {np.sum(partial_success) / num_misplaced:.2f}"
                f" initial misplaced {num_misplaced / len(partial_success):.2f}"
            )

            for _ in range(skip_between_tasks):
                k += 1
                try:
                    next(task_sampler.task_spec_iterator)
                except:
                    return

    init_logging("debug")
    test(
        stage="mini_val_consolidated",
        skip_between_tasks=0,
        random_action_prob=0.0,
        explore=False,
        include_other_move_actions=True,
    )
