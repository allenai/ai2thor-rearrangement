import math

import torch.cuda

from baseline_configs.one_phase.procthor.one_phase_rgb_il_base import (
    OnePhaseRGBILBaseExperimentConfig as Base,
)


class ProcThorOnePhaseRGBClipResNet50DaggerTrainMultiNodeConfig(Base):
    def __init__(
        self,
        distributed_nodes: int = 6,
        il_pipeline_type: str = "40proc",
        expert_exploration_enabled: bool = False,
        include_other_move_actions: bool = True,
        screen_size: int = 224,
        training_steps: int = int(1e9),
        square_root_scaling: bool = True,
        gru_layers: int = 1,
    ):
        self.distributed_nodes = distributed_nodes
        self.IL_PIPELINE_TYPE = il_pipeline_type
        self.EXPERT_EXPLORATION_ENABLED = expert_exploration_enabled
        self.INCLUDE_OTHER_MOVE_ACTIONS = include_other_move_actions
        self.SCREEN_SIZE = screen_size
        self.THOR_CONTROLLER_KWARGS = {
            "rotateStepDegrees": 90,
            "snapToGrid": True,
            "quality": "Very Low",
            "width": screen_size,
            "height": screen_size,
            "fastActionEmit": True,
            "scene": "Procedural",
        }
        self.TRAINING_STEPS = training_steps
        self.square_root_scaling = square_root_scaling
        self.GRU_LAYERS = gru_layers

    def tag(self) -> str:
        return f"ProcThorOnePhaseRGBClipResNet50DaggerTrainNodes{self.distributed_nodes}_{self.IL_PIPELINE_TYPE}_{self.TRAINING_STEPS // int(1e6)}Msteps_{self.GRU_LAYERS}gru_layers"

    def machine_params(self, mode="train", **kwargs):
        params = super().machine_params(mode, **kwargs)

        train_gpu_ids = list(range(torch.cuda.device_count()))

        if mode == "train":
            params.devices = params.devices * self.distributed_nodes
            params.nprocesses = params.nprocesses * self.distributed_nodes
            if params.sampler_devices is not None:
                params.sampler_devices = params.sampler_devices * self.distributed_nodes

            if "machine_id" in kwargs:
                machine_id = kwargs["machine_id"]
                assert (
                    0 <= machine_id < self.distributed_nodes
                ), f"machine_id {machine_id} out of range [0, {self.distributed_nodes - 1}]"

                local_worker_ids = list(
                    range(
                        len(train_gpu_ids) * machine_id,
                        len(train_gpu_ids) * (machine_id + 1),
                    )
                )

                params.set_local_worker_ids(local_worker_ids)

            # Confirm we're setting up train params nicely:
            if "machine_id" in kwargs:
                print(
                    f"devices {params.devices}"
                    f"\nnprocesses {params.nprocesses}"
                    f"\nsampler_devices {params.sampler_devices}"
                    f"\nlocal_worker_ids {params.local_worker_ids}"
                )
        elif mode == "valid":
            # Use all GPUs at their maximum capacity for training
            # (you may run validation in a separate machine)
            params.nprocesses = (0,)

        return params

    def _use_label_to_get_training_params(self):
        params = super()._use_label_to_get_training_params()
        if self.square_root_scaling:
            params["lr"] *= math.sqrt(self.distributed_nodes)  # linear scaling
        else:
            params["lr"] *= self.distributed_nodes  # linear scaling
        return params
