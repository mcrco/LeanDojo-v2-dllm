"""
Usage: python examples/theorem_proving/diffusion_whole_proof.py

Example script for generating a whole proof with the diffusion prover.
"""

from lean_dojo_v2.prover import DiffusionProver

theorem = "theorem my_and_comm : ∀ {p q : Prop}, And p q → And q p := by"
prover = DiffusionProver()
proof = prover.generate_whole_proof(theorem)

print(proof)
