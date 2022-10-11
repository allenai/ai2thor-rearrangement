from abc import ABC
from typing import Optional, Sequence, Dict, Any
import math

import torch.cuda
from allenact.utils.experiment_utils import PipelineStage
from allenact.utils.system import get_logger
from allenact.base_abstractions.sensor import Sensor, ExpertActionSensor

from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
from baseline_configs.one_phase.one_phase_rgb_il_base import (
    Imitation,
    StepwiseLinearDecay,
)

from baseline_configs.one_phase.procthor.one_phase_rgb_base import (
    OnePhaseRGBBaseExperimentConfig,
)


def il_training_params(label: str, training_steps: int, square_root_scaling=False):
    num_train_processes = int(label.split("proc")[0])
    num_steps = 64
    num_mini_batch = 2 if torch.cuda.device_count() > 0 else 1
    prop = (num_train_processes / 40) * (num_steps / 64)  # / num_mini_batch
    if not square_root_scaling:
        lr = 3e-4 * prop
    else:
        lr = 3e-4 * min(math.sqrt(prop), prop)
    update_repeats = 3
    dagger_steps = min(int(2e6), training_steps // 10)
    bc_tf1_steps = min(int(2e5), training_steps // 10)

    get_logger().info(
        f"Using {training_steps // int(1e6)}M training steps and"
        f" {dagger_steps // int(1e6)}M Dagger steps,"
        f" {bc_tf1_steps // int(1e5)}00k BC with teacher forcing=1,"
        f" {num_train_processes} processes (per machine)",
    )

    return dict(
        lr=lr,
        num_steps=num_steps,
        num_mini_batch=num_mini_batch,
        update_repeats=update_repeats,
        use_lr_decay=False,
        num_train_processes=num_train_processes,
        dagger_steps=dagger_steps,
        bc_tf1_steps=bc_tf1_steps,
    )


class OnePhaseRGBILBaseExperimentConfig(OnePhaseRGBBaseExperimentConfig, ABC):
    IL_PIPELINE_TYPE: Optional[str] = None
    square_root_scaling = False

    def sensors(self) -> Sequence[Sensor]:
        return [
            ExpertActionSensor(len(RearrangeBaseExperimentConfig.actions())),
            *super().sensors(),
        ]

    def _training_pipeline_info(self, **kwargs) -> Dict[str, Any]:
        """Define how the model trains."""

        training_steps = self.TRAINING_STEPS
        params = self._use_label_to_get_training_params()
        bc_tf1_steps = params["bc_tf1_steps"]
        dagger_steps = params["dagger_steps"]

        return dict(
            named_losses=dict(imitation_loss=Imitation()),
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss"],
                    max_stage_steps=training_steps,
                    teacher_forcing=StepwiseLinearDecay(
                        cumm_steps_and_values=[
                            (bc_tf1_steps, 1.0),
                            (bc_tf1_steps + dagger_steps, 0.0),
                        ]
                    ),
                )
            ],
            **params,
        )

    def num_train_processes(self) -> int:
        return self._use_label_to_get_training_params()["num_train_processes"]

    def _use_label_to_get_training_params(self):
        return il_training_params(
            label=self.IL_PIPELINE_TYPE.lower(),
            training_steps=self.TRAINING_STEPS,
            square_root_scaling=self.square_root_scaling,
        )
