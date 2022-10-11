from allenact.utils.experiment_utils import (
    PipelineStage,
    TrainingSettings,
)
from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation

from baseline_configs.one_phase.one_phase_rgb_il_base import (
    OnePhaseRGBILBaseExperimentConfig,
)


def il_finetuning_params(label: str):
    if label == "40proc":
        return dict(
            lr=3e-4,
            rollout_steps=[64, 96, 128],
            num_mini_batch=1,
            update_repeats=3,
            num_train_processes=40,
            stage_steps=[int(3e6), int(6e6)],
            use_lr_decay=False,
        )
    else:
        raise NotImplementedError


class OnePhaseRGBClipResNet50FineTuneExperimentConfig(
    OnePhaseRGBILBaseExperimentConfig
):
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = (
        "RN50",
        "clip",
    )
    IL_PIPELINE_TYPE = "40proc"
    TRAINING_STEPS = int(100e6)
    INCLUDE_OTHER_MOVE_ACTIONS = True

    @classmethod
    def tag(self) -> str:
        return f"iThorOnePhaseRGBClipResNet50FineTune"

    @classmethod
    def _use_label_to_get_training_params(cls):
        return il_finetuning_params(label=cls.IL_PIPELINE_TYPE.lower())

    @classmethod
    def _training_pipeline_info(self, **kwargs):
        """Define how the model trains."""

        training_steps = self.TRAINING_STEPS
        params = self._use_label_to_get_training_params()
        rollout_steps = params["rollout_steps"]
        stage_steps = params["stage_steps"] + [
            training_steps - sum(params["stage_steps"])
        ]

        return dict(
            named_losses=dict(imitation_loss=Imitation()),
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss"],
                    max_stage_steps=stage_steps[it],
                    training_settings=TrainingSettings(num_steps=rollout_steps[it]),
                )
                for it in range(len(rollout_steps))
            ],
            num_steps=max(rollout_steps),
            **params,
        )
