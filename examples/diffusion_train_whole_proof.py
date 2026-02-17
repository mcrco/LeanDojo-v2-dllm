"""Train a baseline whole-proof objective from reconstructed tactic scripts."""

from lean_dojo_v2.agent import BaseAgent
from lean_dojo_v2.trainer import DiffusionTrainer


class _DiffusionWholeProofTrainingAgent(BaseAgent):
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
        output_dir="outputs-diffusion-whole-proof",
        objective="whole_proof",
        epochs_per_repo=1,
        batch_size=1,
        lr=2e-5,
        include_focused_tactics=False,
    )

    agent = _DiffusionWholeProofTrainingAgent(trainer=trainer)
    agent.setup_github_repository(url=url, commit=commit)
    agent.train()


if __name__ == "__main__":
    main()
