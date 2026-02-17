"""Demo for prefix/suffix `sorry` infilling with DiffusionProver."""

from lean_dojo_v2.prover import DiffusionProver

prefix = """theorem my_and_comm : ∀ {p q : Prop}, And p q → And q p := by\n  intro p q h\n"""
suffix = "\n"

prover = DiffusionProver()
infilled = prover.infill_sorry(prefix=prefix, suffix=suffix)
print(infilled)
