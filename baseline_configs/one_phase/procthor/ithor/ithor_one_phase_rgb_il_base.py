from abc import ABC
from typing import Tuple, Sequence, Optional, Dict, Any

import torch

from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from allenact.base_abstractions.sensor import ExpertActionSensor, Sensor
from allenact.utils.experiment_utils import PipelineStage
from allenact.utils.misc_utils import all_unique
from baseline_configs.one_phase.procthor.ithor.ithor_one_phase_rgb_base import (
    OnePhaseRGBBaseExperimentConfig,
)


class StepwiseLinearDecay:
    def __init__(self, cumm_steps_and_values: Sequence[Tuple[int, float]]):
        assert len(cumm_steps_and_values) >= 1

        self.steps_and_values = list(sorted(cumm_steps_and_values))
        self.steps = [steps for steps, _ in cumm_steps_and_values]
        self.values = [value for _, value in cumm_steps_and_values]

        assert all_unique(self.steps)
        assert all(0 <= v <= 1 for v in self.values)

    def __call__(self, epoch: int) -> float:
        """Get the value for the input number of steps."""
        if epoch <= self.steps[0]:
            return self.values[0]
        elif epoch >= self.steps[-1]:
            return self.values[-1]
        else:
            for i, (s0, s1) in enumerate(zip(self.steps[:-1], self.steps[1:])):
                if epoch < s1:
                    p = (epoch - s0) / (s1 - s0)
                    v0 = self.values[i]
                    v1 = self.values[i + 1]
                    return p * v1 + (1 - p) * v0


def il_training_params(label: str, training_steps: int):
    use_lr_decay = False

    if label == "80proc":
        lr = 3e-4
        num_train_processes = 80
        num_steps = 64
        dagger_steps = min(int(1e6), training_steps // 10)
        bc_tf1_steps = min(int(1e5), training_steps // 10)
        update_repeats = 3
        num_mini_batch = 2 if torch.cuda.is_available() else 1

    elif label == "40proc":
        lr = 3e-4
        num_train_processes = 40
        num_steps = 64
        dagger_steps = min(int(1e6), training_steps // 10)
        bc_tf1_steps = min(int(1e5), training_steps // 10)
        update_repeats = 3
        num_mini_batch = 1

    elif label == "40proc-longtf":
        lr = 3e-4
        num_train_processes = 40
        num_steps = 64
        dagger_steps = min(int(5e6), training_steps // 10)
        bc_tf1_steps = min(int(5e5), training_steps // 10)
        update_repeats = 3
        num_mini_batch = 1

    else:
        raise NotImplementedError

    return dict(
        lr=lr,
        num_steps=num_steps,
        num_mini_batch=num_mini_batch,
        update_repeats=update_repeats,
        use_lr_decay=use_lr_decay,
        num_train_processes=num_train_processes,
        dagger_steps=dagger_steps,
        bc_tf1_steps=bc_tf1_steps,
    )


class OnePhaseRGBILBaseExperimentConfig(OnePhaseRGBBaseExperimentConfig, ABC):
    IL_PIPELINE_TYPE: Optional[str] = None

    def sensors(self) -> Sequence[Sensor]:
        return [
            *super(OnePhaseRGBILBaseExperimentConfig, self).sensors(),
            ExpertActionSensor(len(self.actions())),
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
            **params
        )

    def num_train_processes(self) -> int:
        return self._use_label_to_get_training_params()["num_train_processes"]

    def _use_label_to_get_training_params(self):
        return il_training_params(
            label=self.IL_PIPELINE_TYPE.lower(), training_steps=self.TRAINING_STEPS
        )
