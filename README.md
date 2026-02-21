# Distinction Structures

**A minimal self-modeling framework for relational description, with machine-checked proofs in Lean 4.**

---

## Three Theorems

This repository contains Lean 4 formalizations of three results about finite algebraic structures that model themselves. All proofs compile with **zero `sorry`** on Lean 4.28.0 / Mathlib v4.28.0.

**Theorem 1 (Existence).** Intrinsically reflexive Distinction Structures exist. A 16-element symmetric algebra (Δ₀) and a 17-element directed algebra (Δ₁) each satisfy axioms A1--A7', Ext, and contain behavioral self-models: internal encodings whose elements, when composed by the structure's own operation, reproduce the behavior of the structure's own components.

**Theorem 2 (Discoverability).** Discoverably reflexive Distinction Structures exist. The 17-element directed model Δ₁ has a self-model that is recoverable from black-box probing alone, with each of the 8 recovery steps proved unique. An observer with no prior knowledge can identify every structural component purely from the operation table.

**Theorem 3 (Irreducibility).** Actuality is not determined by structure. Two models (Δ₁ and Δ₁') on the same 18-element carrier share 322 out of 324 operation table entries, both satisfy all axioms and reflexivity conditions, yet differ in actuality assignment. No structural predicate resolves the difference. The only way to determine which elements are actual is to query the actuality tester directly.

Three machine-checked results. Self-description is possible. Communication is possible. But the question of what's real cannot be settled by structure alone.

---

## Why It Matters

Many systems can represent themselves (Godel numbering, quines, metacircular evaluators). Fewer do so *behaviorally* -- where the encoding elements don't just name components but act like them under the system's own operation. Fewer still are *discoverable* -- where an external observer can recover the self-model with no documentation.

Δ₁ achieves all three: self-representation, behavioral fidelity, and black-box recoverability. The recovery procedure is not a heuristic -- each step is a uniqueness lemma, machine-checked over the finite domain.

The irreducibility result shows what the framework *cannot* do. Given a complete structural description of a self-modeling system, the question "which elements are actual?" has multiple valid answers, and the structure alone does not select among them. Two fully valid self-modeling Distinction Structures can agree on every compositional fact and disagree only on actuality. The actuality tester carries irreducible information: there is no structural back door.

This is "existence is not a predicate" as a machine-checked theorem. Not as a philosophical argument, not as an interpretation, not as a slogan -- as a Lean theorem that compiles with zero sorry.

---

## Repository Structure

```
DistinctionStructures/
├── lakefile.lean                                # Build configuration
├── lean-toolchain                               # Lean version pin
├── DistinctionStructures/
│   ├── Basic.lean                               # Abstract DS definitions and axioms
│   ├── Delta0.lean                              # Δ₀: 16-element symmetric model
│   ├── Delta1.lean                              # Δ₁: 17-element directed model
│   ├── Discoverable.lean                        # 8 recovery lemmas (discoverability)
│   └── ActualityIrreducibility.lean             # Actuality irreducibility theorem
├── python/
│   └── delta2_interpreter.py                    # Δ₂: interpreter with QUOTE/EVAL/APP/UNAPP
├── docs/
│   ├── Distinction_Structures.md                # Full document with proofs and philosophy
│   ├── ARTIFACT.md                              # Artifact guide: what is proved and how
│   └── COMMUNICATION.md                         # Communication protocol design
└── README.md
```

## Building

```bash
# Requires Lean 4.28.0 and Mathlib v4.28.0
lake build
```

All theorems are checked by `decide` or `native_decide`, which is appropriate and complete for finite carrier types with decidable equality. The full project is ~1270 lines of Lean. See [ARTIFACT.md](docs/ARTIFACT.md) for details.

---

## The Three Results in Detail

### Theorem 1: Existence (Δ₀ and Δ₁)

| Property | Δ₀ (Symmetric) | Δ₁ (Directed) |
|----------|----------------|----------------|
| Elements in D(ι) | 14 (+ 2 in D(κ)) | 17 (+ 2 in D(κ)) |
| Operation | Σ : Finset D(ι) → D(ι) | · : D(ι) → D(ι) → D(ι) |
| Rules | Priority-based on Finset | 26 first-match rules |
| Self-model | 12 encoding elements | Encoding elements with H1--H3 |
| Reflexivity | Intrinsic | Discoverable |
| Lean files | `Delta0.lean` | `Delta1.lean` + `Discoverable.lean` |

Both models satisfy A1--A7', Ext, H1--H3, and IR1--IR5. The switch from symmetric to directed composition costs one additional element but unlocks discoverability.

### Theorem 2: Discoverability (Recovery Lemmas)

An observer with access only to the operation `dot : D → D → D` (no documentation, no labels) can recover every structural component of Δ₁:

| Step | What is recovered | Lean theorem |
|------|-------------------|-------------|
| 1 | Booleans (top, bot) -- the only left-absorbers | `boolean_uniqueness` |
| 2 | Testers (e_I, d_K, m_K, m_I) -- non-booleans with boolean-valued output | `tester_characterization` |
| 3 | Tester signatures by decoded-set size (16, 2, 2, 1) | `tester_card_m_I`, `tester_card_e_I`, `tester_card_d_K`, `tester_card_m_K` |
| 4 | Context tester vs domain tester -- rich vs inert decoded elements | `rich_context_tokens`, `inert_kappa_tokens` |
| 5 | e_D vs e_M -- encoder asymmetry | `encoder_pair`, `encoder_asymmetry` |
| 6 | i vs k -- context token discrimination | `context_token_discrimination` |
| 7 | p (non-actual element) -- unique m_I reject | `junk_identification` |
| 8 | e_Sigma, s_C, e_Delta -- unique triple with synthesis property | `triple_identification`, `triple_uniqueness` |

Each step has exactly one solution. The recovery is not heuristic -- it is mechanically verified.

### Theorem 3: Actuality Irreducibility

Two models, Δ₁ and Δ₁', are constructed on the same 18-element carrier (the 17 elements of Δ₁ plus a second surplus element q). They differ in exactly two entries of the 18x18 operation table -- both in the m_I row:

| | Δ₁ | Δ₁' |
|---|---|---|
| m_I · p | bot (rejects p) | top (accepts p) |
| m_I · q | top (accepts q) | bot (rejects q) |
| All other 322 entries | identical | identical |
| Actuality set | M = D \ {p} | M' = D \ {q} |

Both models independently satisfy:
- Ext (behavioral separability)
- H1, H2, H3 (homomorphism conditions)
- IR1, IR2, IR4 (intrinsic reflexivity conditions)
- A2, A5, A7' (existence, selectivity, structural novelty)

Key theorems in `ActualityIrreducibility.lean`:

| Theorem | What it proves |
|---------|---------------|
| `ops_agree_non_mI` | ∀ x y, x ≠ m_I → dot1 x y = dot1' x y |
| `ops_differ_only_mI` | The only disagreements are at (m_I, p) and (m_I, q) |
| `right_image_same_dot1` | ∀ x ≠ m_I, dot1 x p = dot1 x q |
| `cross_model_right_image` | ∀ x ≠ m_I, dot1 x p = dot1' x q |
| `mI_is_unique_discriminator` | m_I is the only element that classifies p and q differently |
| `no_universal_actuality_predicate` | No predicate matches actualM in Δ₁ and actualM' in Δ₁' |
| `actuality_irreducibility` | Combined 7-conjunct theorem |

The `cross_model_right_image` result is the sharp version: non-m_I elements cannot distinguish p from q at all as right arguments. The only element that "knows" which one is non-actual is m_I itself -- and m_I's behavior is the one thing that differs between the models. The only way to determine what's actual is to already have an actuality predicate. There is no structural back door.

This connects directly to epistemology: given a complete structural description of a self-modeling system, the question "which elements are actual?" has multiple valid answers, and the structure alone does not select among them. You have to look. You have to encounter which model you're in. No amount of reasoning about the structure resolves it.

---

## Δ₂ -- Computational Extension (Interpreter)

| Property | Value |
|----------|-------|
| Core atoms | 21 (17 from Δ₁ + QUOTE, EVAL, APP, UNAPP) |
| Data domain | Unbounded (quoted terms, application nodes) |
| Operation | Recursive interpreter extending Δ₁'s dot |
| Status | Python implementation, not Lean-formalized |
| File | `python/delta2_interpreter.py` |

Δ₂ is not a finite algebra. It is a Distinction Structure core embedded in an interpreter. QUOTE generates unbounded inert values; EVAL is defined recursively over syntax trees. This is the boundary between algebra and computation.

## What Is Not Proved

- **Minimality.** We do not prove that 16 (resp. 17) is the minimum element count. The models are upper bound witnesses.
- **Symmetric impossibility.** The symmetric synthesis barrier is demonstrated by construction but not proved as a general impossibility theorem.
- **Categorical formalization.** The category-theoretic perspective is discussed in the document but not formalized in Lean.
- **Δ₂ properties.** The computational extension is implemented in Python but not mechanically verified.

## Communication Protocol

The discoverability property has a direct application: a communication protocol for unknown intelligences that requires no prior shared language.

The protocol has four layers:

- **Layer A** -- The Cayley table (self-interpreting, medium-independent)
- **Layer B** -- Medium-reflexive grounding (anchors vocabulary in the transmission medium itself)
- **Layer C** -- Extended physics (new domains introduced via the encoder apparatus)
- **Layer D** -- Open communication (executable programs via Δ₂)

The key innovation is that Layer B references the transmission medium -- the one physical context sender and recipient provably share. See [`COMMUNICATION.md`](COMMUNICATION.md) for the full protocol design.

## Background Document

The full mathematical and philosophical development is in [`docs/Distinction_Structures.md`](docs/Distinction_Structures.md). It covers the four concepts (Distinction, Context, Actuality, Synthesis), both existence proofs, the recovery procedure, the symmetric synthesis barrier, epistemological implications, and the path to computation.

## License

MIT

## Citation

If you use this work, please cite:

```
@software{distinction_structures_2026,
  author = {Stefano Palmieri},
  title = {Distinction Structures: A Minimal Self-Modeling Framework},
  year = {2026},
  note = {Lean 4 formalization, 0 sorry. Three machine-checked results: existence (Δ₀, Δ₁), discoverability (8 recovery lemmas), and actuality irreducibility.}
}
```
