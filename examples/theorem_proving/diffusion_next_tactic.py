"""
Usage: python examples/theorem_proving/diffusion_next_tactic.py

Example script for proving a theorem with the diffusion prover using proof search.
"""

from pantograph.server import Server

from lean_dojo_v2.prover import DiffusionProver

server = Server()
prover = DiffusionProver()

result, used_tactics = prover.search(
    server=server, goal="∀ {p q : Prop}, p ∧ q → q ∧ p", verbose=False
)

print(result)
if result.success:
    for tactic in used_tactics:
        print(tactic)
