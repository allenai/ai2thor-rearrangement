from typing import Dict, Union, List, Any, Optional, Sequence

import ai2thor.controller
import lru
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_util import include_object_data

from rearrange.utils import hand_in_initial_position

_UNIFORM_BOX_CACHE = {}


class ObjectInteractablePostionsCache:
    def __init__(self, max_size: int = 20000, ndigits=2):
        self._key_to_positions = lru.LRU(size=max_size)

        self.ndigits = ndigits
        self.max_size = max_size

    def reset_cache(self):
        self._key_to_positions.clear()

    def _get_key(
        self,
        scene_name: str,
        obj: Dict[str, Any],
        hor: Optional[float],
        stand: Optional[bool],
    ):
        p = obj["position"]
        return (
            scene_name,
            obj["type"] if "type" in obj else obj["objectType"],
            round(p["x"], self.ndigits),
            round(p["y"], self.ndigits),
            round(p["z"], self.ndigits),
            hor,
            stand,
        )

    def get(
        self,
        scene_name: str,
        obj: Dict[str, Any],
        controller: ai2thor.controller.Controller,
        reachable_positions: Optional[Sequence[Dict[str, float]]] = None,
        force_cache_refresh: bool = False,
        force_horizon: Optional[int] = None,
        force_standing: Optional[bool] = None,
        avoid_teleport: bool = False,
    ) -> List[Dict[str, Union[float, int, bool]]]:
        scene_name = scene_name.replace("_physics", "")
        obj_key = self._get_key(
            scene_name=scene_name, obj=obj, hor=force_horizon, stand=force_standing
        )

        env = None
        if hasattr(controller, "controller"):
            env = controller
            controller = env.controller

        if force_cache_refresh or obj_key not in self._key_to_positions:
            with include_object_data(controller):
                metadata = controller.last_event.metadata

            if env is None:
                cur_scene_name = metadata["sceneName"].replace("_physics", "")
            else:
                cur_scene_name = env.scene
            assert (
                scene_name == cur_scene_name
            ), f"Scene names must match when filling a cache miss ({scene_name} != {cur_scene_name})."

            obj_in_scene = next(
                (o for o in metadata["objects"] if o["objectId"] == obj["objectId"]),
                None,
            )
            if obj_in_scene is None:
                raise RuntimeError(
                    f"Object with name {obj['objectId']} must be in the scene when filling a cache miss"
                )

            desired_pos = obj["position"]
            desired_rot = obj["rotation"]

            cur_pos = obj_in_scene["position"]
            cur_rot = obj_in_scene["rotation"]

            should_teleport = (
                IThorEnvironment.position_dist(desired_pos, cur_pos) >= 1e-3
                or IThorEnvironment.rotation_dist(desired_rot, cur_rot) >= 1
            ) and not avoid_teleport

            object_held = obj_in_scene["isPickedUp"]
            physics_was_unpaused = controller.last_event.metadata.get(
                "physicsAutoSimulation", True
            )
            if should_teleport:
                if object_held:
                    if not hand_in_initial_position(
                        controller=controller, ignore_rotation=True
                    ):
                        raise NotImplementedError

                    if physics_was_unpaused:
                        controller.step("PausePhysicsAutoSim")
                        assert controller.last_event.metadata["lastActionSuccess"]

                event = controller.step(
                    "TeleportObject",
                    objectId=obj_in_scene["objectId"],
                    rotation=desired_rot,
                    **desired_pos,
                    forceAction=True,
                    allowTeleportOutOfHand=True,
                    forceKinematic=True,
                )
                assert event.metadata["lastActionSuccess"]

            options = {}
            if force_standing is not None:
                options["standings"] = [force_standing]
            if force_horizon is not None:
                options["horizons"] = [force_horizon]

            metadata = controller.step(
                action="GetInteractablePoses",
                objectId=obj["objectId"],
                positions=reachable_positions,
                **options,
            ).metadata
            assert metadata["lastActionSuccess"]
            self._key_to_positions[obj_key] = metadata["actionReturn"]

            if should_teleport:
                if object_held:
                    if hand_in_initial_position(
                        controller=controller, ignore_rotation=True
                    ):
                        controller.step(
                            "PickupObject",
                            objectId=obj_in_scene["objectId"],
                            forceAction=True,
                        )
                        assert controller.last_event.metadata["lastActionSuccess"]

                        if physics_was_unpaused:
                            controller.step("UnpausePhysicsAutoSim")
                            assert controller.last_event.metadata["lastActionSuccess"]
                    else:
                        raise NotImplementedError
                else:
                    event = controller.step(
                        "TeleportObject",
                        objectId=obj_in_scene["objectId"],
                        rotation=cur_rot,
                        **cur_pos,
                        forceAction=True,
                    )
                    assert event.metadata["lastActionSuccess"]

        return self._key_to_positions[obj_key]
