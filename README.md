# Distinction Structures

**A minimal self-modeling framework for relational description, with machine-checked proofs in Lean 4.**

---

## What This Is

This repository contains Lean 4 formalizations of two theorems about finite algebraic structures that model themselves:

**Theorem (Δ₀).** There exists a 16-element symmetric algebra satisfying axioms A1–A7′ and Ext that contains a behavioral self-model — an internal encoding whose elements, when composed by the structure's own operation, reproduce the behavior of the structure's own components.

**Theorem (Δ₁).** There exists a 17-element directed algebra satisfying the same axioms (adapted for directed composition) whose self-model is *discoverable*: an observer with no prior knowledge can recover the complete encoding purely by probing the operation, with each recovery step proved unique.

**Cost bound.** Discoverable reflexivity requires at most one additional element and the switch from symmetric to directed composition, compared to intrinsic (non-discoverable) reflexivity.

All proofs compile with **zero `sorry`** on Lean 4.28.0 / Mathlib v4.28.0.

## Why It Matters

Many systems can represent themselves (Gödel numbering, quines, metacircular evaluators). Fewer do so *behaviorally* — where the encoding elements don't just name components but act like them under the system's own operation. Fewer still are *discoverable* — where an external observer can recover the self-model with no documentation.

Δ₁ achieves all three: self-representation, behavioral fidelity, and black-box recoverability. The recovery procedure is not a heuristic — each step is a uniqueness lemma, machine-checked over the finite domain.

The framework also reveals a structural barrier: **symmetric composition supports self-modeling but obstructs self-announcement**. To make a self-model externally legible, directed (ordered) composition is required. This is a formal analogue of the function/argument distinction in logic and the speaker/listener asymmetry in communication.

## Repository Structure

```
DistinctionStructures/
├── lakefile.lean                          # Build configuration
├── lean-toolchain                         # Lean version pin
├── DistinctionStructures/
│   ├── Basic.lean                         # Abstract DS definitions and axioms
│   ├── Directed.lean                      # Directed DS definitions
│   ├── Delta0.lean                        # Δ₀: 16-element symmetric model
│   ├── Delta1.lean                        # Δ₁: 17-element directed model
│   └── Discoverable.lean                  # 8 recovery lemmas (discoverability proof)
├── python/
│   └── delta2_interpreter.py              # Δ₂: interpreter with QUOTE/EVAL/APP/UNAPP
├── docs/
│   ├── Distinction_Structures.md          # Full document with proofs and philosophy
│   ├── ARTIFACT.md                        # Artifact guide: what is proved and how
│   └── COMMUNICATION.md                   # Communication protocol for unknown intelligences
└── README.md
```

## Building

```bash
# Requires Lean 4.28.0 and Mathlib v4.28.0
lake build
```

All theorems are checked by `decide` or `native_decide`, which is appropriate and complete for finite carrier types with decidable equality. See [ARTIFACT.md](docs/ARTIFACT.md) for details.

## The Two Models

### Δ₀ — Intrinsic Reflexivity (Symmetric)

| Property | Value |
|----------|-------|
| Elements in D(ι) | 14 (+ 2 in D(κ)) |
| Total elements | 16 |
| Operation | Σ : Finset D(ι) → D(ι) (symmetric, set-based) |
| Self-model | 12 encoding elements with H1–H3 verified |
| Reflexivity level | Intrinsic (behavioral, not discoverable) |
| Lean file | `Delta0.lean` |

### Δ₁ — Discoverable Reflexivity (Directed)

| Property | Value |
|----------|-------|
| Elements in D(ι) | 17 (+ 2 in D(κ)) |
| Total elements | 19 |
| Operation | · : D(ι) → D(ι) → D(ι) (directed, ordered) |
| Rules | 26 (first-match priority) |
| Self-model | Encoding elements with H1–H3 verified |
| Reflexivity level | Discoverable (recoverable from black-box probing) |
| Recovery steps | 8, each with uniqueness proof |
| Lean file | `Delta1.lean` + `Discoverable.lean` |

### Δ₂ — Computational Extension (Interpreter)

| Property | Value |
|----------|-------|
| Core atoms | 21 (17 from Δ₁ + QUOTE, EVAL, APP, UNAPP) |
| Data domain | Unbounded (quoted terms, application nodes) |
| Operation | Recursive interpreter extending Δ₁'s dot |
| Status | Python implementation, not Lean-formalized |
| File | `python/delta2_interpreter.py` |

Δ₂ is **not** a finite algebra. It is a Distinction Structure core embedded in an interpreter. QUOTE generates unbounded inert values; EVAL is defined recursively over syntax trees. This is the boundary between algebra and computation.

## Key Results (with Lean theorem names)

### Axiom Satisfaction

| Axiom | Δ₀ theorem | Δ₁ theorem |
|-------|-----------|-----------|
| Ext (behavioral separability) | `Delta0.ext_Dι` | `Delta1.ext_D1ι` |
| A7′ (structural novelty) | `Delta0.a7'` | `Delta1.a7'` |
| H1 (Distinction homomorphism) | `Delta0.h1_iota`, `Delta0.h1_kappa` | `Delta1.h1_iota`, `Delta1.h1_kappa` |
| H2 (Actuality homomorphism) | `Delta0.h2_iota`, `Delta0.h2_kappa` | `Delta1.h2_iota`, `Delta1.h2_kappa` |
| H3 (Synthesis homomorphism) | `Delta0.h3` | `Delta1.h3` |

### Recovery Lemmas (Δ₁ only)

| Step | What is recovered | Lean theorem |
|------|-------------------|-------------|
| 1 | Booleans (⊤, ⊥) | `Discoverable.boolean_uniqueness` |
| 2 | Testers (e_I, d_K, m_K, m_I) | `Discoverable.tester_characterization` |
| 3 | Tester signatures (16, 2, 2, 1) | `Discoverable.tester_cardinality_*` |
| 4 | Context tester vs domain tester | `Discoverable.rich_vs_inert` |
| 5 | e_D vs e_M | `Discoverable.encoder_asymmetry` |
| 6 | i vs k | `Discoverable.context_token_discrimination` |
| 7 | p (non-actual element) | `Discoverable.junk_identification` |
| 8 | e_Σ, s_C, e_Δ (unique triple) | `Discoverable.triple_identification` |

## What Is Not Proved

- **Minimality.** We do not prove that 16 (resp. 17) is the minimum element count. The models are upper bound witnesses.
- **Symmetric impossibility.** The symmetric synthesis barrier is demonstrated by construction (boolean contradiction, operator/operand conflict) but not proved as a general impossibility theorem.
- **Categorical formalization.** The category-theoretic perspective (Contexts as maximal cliques, Actuality as sub-presheaf) is discussed in the document but not formalized in Lean.
- **Δ₂ properties.** The computational extension is implemented in Python but not mechanically verified.

## Background Document

The full mathematical and philosophical development is in [`docs/Distinction_Structures.md`](docs/Distinction_Structures.md). It covers:

- The four concepts (Distinction, Context, Actuality, Synthesis) and their philosophical motivation
- Set-theoretic and category-theoretic formalizations
- Both existence proofs with complete operation tables
- The symmetric synthesis barrier
- The recovery procedure with all uniqueness arguments
- Epistemological implications
- Computational interpretation (path to programming language)
- Status of all claims (proven / conjectured / retracted)

## Communication Protocol

The discoverability property has a direct application: a communication protocol for unknown intelligences that requires no prior shared language.

The protocol has four layers:

- **Layer A** — The Cayley table (self-interpreting, medium-independent)
- **Layer B** — Medium-reflexive grounding (anchors vocabulary in the transmission medium itself)
- **Layer C** — Extended physics (new domains introduced via the encoder apparatus)
- **Layer D** — Open communication (executable programs via Δ₂)

The key innovation is that Layer B references the transmission medium — the one physical context sender and recipient provably share. See [`docs/COMMUNICATION.md`](docs/COMMUNICATION.md) for the full protocol design.

## License

MIT

## Citation

If you use this work, please cite:

```
@software{distinction_structures_2026,
  author = {Stefano Palmieri},
  title = {Distinction Structures: A Minimal Self-Modeling Framework},
  year = {2026},
  note = {Lean 4 formalization, 0 sorry. Models Δ₀ (intrinsic reflexivity) and Δ₁ (discoverable reflexivity) machine-checked.}
}
```
