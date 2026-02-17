"""Example entrypoint for running DiffusionAgent on a traced GitHub repo."""

from lean_dojo_v2.agent import DiffusionAgent


def main():
    url = "https://github.com/durant42040/lean4-example"
    commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

    agent = DiffusionAgent()
    agent.setup_github_repository(url=url, commit=commit)
    agent.prove()


if __name__ == "__main__":
    main()
