"""Train the diffusion denoising objective O2 (tactic-script infilling)."""

from lean_dojo_v2.agent import BaseAgent
from lean_dojo_v2.trainer import DiffusionTrainer


class _DiffusionInfillTrainingAgent(BaseAgent):
    def __init__(self, trainer: DiffusionTrainer):
        super().__init__()
        self.trainer = trainer

    def _get_build_deps(self) -> bool:
        return False

    def _setup_prover(self):
        raise NotImplementedError("Training-only helper agent")


def main():
    url = "https://github.com/durant42040/lean4-example"
    commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

    trainer = DiffusionTrainer(
        model_name="inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        output_dir="outputs-diffusion-infill",
        objective="infill",
        epochs_per_repo=1,
        batch_size=1,
        lr=2e-5,
        num_holes=1,
        max_hole_len=3,
    )

    agent = _DiffusionInfillTrainingAgent(trainer=trainer)
    agent.setup_github_repository(url=url, commit=commit)
    agent.train()


if __name__ == "__main__":
    main()
