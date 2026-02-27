#!/usr/bin/env python3
"""
Δ₂+Byte — Byte Arithmetic Extension of the Δ₂ Algebra

Extends the 21-atom Δ₂ algebra with 34 new atoms:
  - 18 byte operations (arithmetic, bitwise, predicates, literal constructor)
  - 16 nibble atoms N0–NF forming Z/16Z under dot

Total: 55 atoms. All atoms remain uniquely recoverable from black-box
access to the dot operation alone.

Byte values are structured terms bv(n) where n ∈ {0..255}.
Multi-arg ops use curried partial application.

Usage:
  python delta_byte.py            # run verification suite
"""

from dataclasses import dataclass
from typing import Any


# ============================================================================
# Runtime term representations
# ============================================================================

@dataclass(frozen=True)
class Atom:
    name: str

@dataclass(frozen=True)
class Quote:
    term: Any

@dataclass(frozen=True)
class AppNode:
    f: Any
    x: Any

@dataclass(frozen=True)
class UnappBundle:
    f: Any
    x: Any

@dataclass(frozen=True)
class Partial:
    op: str
    a: Any

@dataclass(frozen=True)
class ByteVal:
    val: int  # 0–255

@dataclass(frozen=True)
class BLitPartial:
    hi: int  # 0–15 (high nibble, waiting for low)

@dataclass(frozen=True)
class ByteOp1Partial:
    """Binary byte op with first argument applied."""
    op: str
    first: int  # 0–255

@dataclass(frozen=True)
class CarryOp1Partial:
    """Ternary carry op with first argument applied."""
    op: str
    first: int

@dataclass(frozen=True)
class CarryOp2Partial:
    """Ternary carry op with two arguments applied, waiting for carry bit."""
    op: str
    first: int
    second: int


def A(n: str) -> Atom:
    return Atom(n)


# ============================================================================
# Atom inventory
# ============================================================================

NAMES_D1 = [
    "⊤", "⊥", "i", "k", "a", "b", "e_I",
    "e_D", "e_M", "e_Σ", "e_Δ",
    "d_I", "d_K", "m_I", "m_K", "s_C", "p",
]
NAMES_D2 = ["QUOTE", "EVAL", "APP", "UNAPP"]
NAMES_BYTE_OPS = [
    "B_ADD", "B_SUB", "B_MUL", "B_DIV", "B_MOD",
    "B_AND", "B_OR", "B_XOR", "B_NOT", "B_SHL", "B_SHR",
    "B_ADC", "B_SBC", "B_ROL",
    "B_ZERO", "B_EQ", "B_LT", "B_LIT",
]
NAMES_NIBBLES = [f"N{i:X}" for i in range(16)]  # N0, N1, ..., NF

ALL_NAMES = NAMES_D1 + NAMES_D2 + NAMES_BYTE_OPS + NAMES_NIBBLES
ALL_ATOMS = [A(n) for n in ALL_NAMES]

NIBBLE_ATOMS = frozenset(A(f"N{i:X}") for i in range(16))
BYTE_OP_ATOMS = frozenset(A(n) for n in NAMES_BYTE_OPS)
NEW_ATOMS = NIBBLE_ATOMS | BYTE_OP_ATOMS  # 34 new atoms
D1_ATOMS = frozenset(A(n) for n in NAMES_D1)
D2_EXT_ATOMS = frozenset(A(n) for n in NAMES_D2)

BINARY_BYTE_OPS = frozenset(A(n) for n in [
    "B_ADD", "B_SUB", "B_MUL", "B_DIV", "B_MOD",
    "B_AND", "B_OR", "B_XOR", "B_SHL", "B_SHR",
    "B_EQ", "B_LT",
])
TERNARY_BYTE_OPS = frozenset(A(n) for n in ["B_ADC", "B_SBC", "B_ROL"])
UNARY_BYTE_OPS = frozenset(A(n) for n in ["B_NOT", "B_ZERO"])


# ============================================================================
# Nibble helpers
# ============================================================================

def is_nibble(x: Any) -> bool:
    return isinstance(x, Atom) and x in NIBBLE_ATOMS

def nibble_val(x: Atom) -> int:
    return int(x.name[1:], 16)

def nibble(n: int) -> Atom:
    return A(f"N{n % 16:X}")


# ============================================================================
# Δ₁ core: directed application on atoms
# ============================================================================

def dot_iota_d1(x: Atom, y: Atom) -> Atom:
    TOP, BOT = A("⊤"), A("⊥")
    if x == TOP: return TOP
    if x == BOT: return BOT
    if x == A("e_I"): return TOP if y in (A("i"), A("k")) else BOT
    if x == A("d_K"): return TOP if y in (A("a"), A("b")) else BOT
    if x == A("m_K"): return TOP if y == A("a") else BOT
    if x == A("m_I"): return BOT if y == A("p") else TOP
    if x == A("e_D") and y == A("i"): return A("d_I")
    if x == A("e_D") and y == A("k"): return A("d_K")
    if x == A("e_M") and y == A("i"): return A("m_I")
    if x == A("e_M") and y == A("k"): return A("m_K")
    if x == A("e_Σ") and y == A("s_C"): return A("e_Δ")
    if x == A("e_Δ") and y == A("e_D"): return A("d_I")
    if x == A("p") and y == TOP: return TOP
    if y == TOP and x in (A("i"), A("k"), A("a"), A("b"), A("d_I"), A("s_C")):
        return x
    return A("p")


# ============================================================================
# Byte operation implementations
# ============================================================================

def _apply_binary_byte_op(op: str, a: int, b: int) -> Any:
    """Apply a binary byte operation, returning ByteVal or Atom."""
    if op == "B_ADD": return ByteVal((a + b) % 256)
    if op == "B_SUB": return ByteVal((a - b) % 256)
    if op == "B_MUL": return ByteVal((a * b) % 256)
    if op == "B_DIV": return ByteVal(a // b) if b != 0 else A("p")
    if op == "B_MOD": return ByteVal(a % b) if b != 0 else A("p")
    if op == "B_AND": return ByteVal(a & b)
    if op == "B_OR":  return ByteVal(a | b)
    if op == "B_XOR": return ByteVal(a ^ b)
    if op == "B_SHL": return ByteVal((a << b) & 0xFF)
    if op == "B_SHR": return ByteVal(a >> b if b < 8 else 0)
    if op == "B_EQ":  return A("⊤") if a == b else A("⊥")
    if op == "B_LT":  return A("⊤") if a < b else A("⊥")
    return A("p")


def _apply_carry_op(op: str, a: int, b: int, carry: int) -> Any:
    """Apply a ternary carry operation. carry ∈ {0, 1}."""
    if op == "B_ADC": return ByteVal((a + b + carry) % 256)
    if op == "B_SBC": return ByteVal((a - b - carry) % 256)
    if op == "B_ROL":
        shift = b % 8
        if shift == 0:
            return ByteVal((a & 0xFE) | carry)
        rotated = ((a << shift) | (a >> (8 - shift))) & 0xFF
        return ByteVal((rotated & 0xFE) | carry)
    return A("p")


# ============================================================================
# Atom-atom Cayley table (extended)
# ============================================================================

def atom_dot(x: Atom, y: Atom) -> Atom:
    """Cayley table for atom × atom interactions."""
    TOP, BOT = A("⊤"), A("⊥")

    # Booleans absorb everything (Δ₁ rule, extends to all atoms)
    if x == TOP: return TOP
    if x == BOT: return BOT

    # Self-identification on k for all 34 new atoms
    if y == A("k") and x in NEW_ATOMS:
        return x

    # Nibble group: Z/16Z under addition
    if is_nibble(x) and is_nibble(y):
        return nibble((nibble_val(x) + nibble_val(y)) % 16)

    # i acts as successor on nibbles
    if x == A("i") and is_nibble(y):
        return nibble((nibble_val(y) + 1) % 16)

    # Δ₁ atoms: use existing Cayley table
    if x in D1_ATOMS and y in D1_ATOMS:
        return dot_iota_d1(x, y)

    # Everything else → p
    return A("p")


# ============================================================================
# Extended dot operation
# ============================================================================

def eval_term(t: Any) -> Any:
    if isinstance(t, Atom): return t
    if isinstance(t, Quote): return t
    if isinstance(t, AppNode):
        return dot_ext(eval_term(t.f), eval_term(t.x))
    return A("p")


def dot_ext(x: Any, y: Any) -> Any:
    """The full Δ₂+Byte operation on terms."""

    # --- Partial applications (inherited from Δ₂) ---
    if isinstance(x, Partial):
        return AppNode(x.a, y) if x.op == "APP" else A("p")

    # --- BLitPartial: waiting for low nibble ---
    if isinstance(x, BLitPartial):
        if isinstance(y, Atom) and is_nibble(y):
            return ByteVal(x.hi * 16 + nibble_val(y))
        return A("p")

    # --- ByteOp1Partial: binary op waiting for second byte ---
    if isinstance(x, ByteOp1Partial):
        if isinstance(y, ByteVal):
            return _apply_binary_byte_op(x.op, x.first, y.val)
        return A("p")

    # --- CarryOp1Partial: ternary op waiting for second byte ---
    if isinstance(x, CarryOp1Partial):
        if isinstance(y, ByteVal):
            return CarryOp2Partial(x.op, x.first, y.val)
        return A("p")

    # --- CarryOp2Partial: ternary op waiting for carry bit ---
    if isinstance(x, CarryOp2Partial):
        if y == A("⊤"):
            return _apply_carry_op(x.op, x.first, x.second, 1)
        if y == A("⊥"):
            return _apply_carry_op(x.op, x.first, x.second, 0)
        return A("p")

    # --- QUOTE (inherited) ---
    if x == A("QUOTE"): return Quote(y)

    # --- APP (inherited) ---
    if x == A("APP"): return Partial("APP", y)

    # --- UNAPP (inherited) ---
    if x == A("UNAPP"):
        return UnappBundle(y.f, y.x) if isinstance(y, AppNode) else A("p")

    # --- Bundle queries (inherited) ---
    if isinstance(x, UnappBundle):
        if y == A("⊤"): return x.f
        if y == A("⊥"): return x.x
        return A("p")

    # --- EVAL (inherited) ---
    if x == A("EVAL"):
        if isinstance(y, Quote): return eval_term(y.term)
        return A("p")

    # --- B_LIT: first application on nibble atom ---
    if x == A("B_LIT"):
        if isinstance(y, Atom) and is_nibble(y):
            return BLitPartial(nibble_val(y))
        return A("p")

    # --- Unary byte ops on byte values ---
    if x == A("B_NOT") and isinstance(y, ByteVal):
        return ByteVal((~y.val) & 0xFF)
    if x == A("B_ZERO") and isinstance(y, ByteVal):
        return A("⊤") if y.val == 0 else A("⊥")

    # --- Binary byte ops: first application on byte value ---
    if isinstance(x, Atom) and x in BINARY_BYTE_OPS and isinstance(y, ByteVal):
        return ByteOp1Partial(x.name, y.val)

    # --- Ternary byte ops: first application on byte value ---
    if isinstance(x, Atom) and x in TERNARY_BYTE_OPS and isinstance(y, ByteVal):
        return CarryOp1Partial(x.name, y.val)

    # --- Atoms acting on non-atom structured terms → p ---
    if isinstance(x, Atom) and not isinstance(y, Atom):
        return A("p")

    # --- Atom × Atom: extended Cayley table ---
    if isinstance(x, Atom) and isinstance(y, Atom):
        return atom_dot(x, y)

    # --- Default ---
    return A("p")


# ============================================================================
# Verification
# ============================================================================

def verify_nibble_group():
    """Verify nibble atoms form Z/16Z under dot."""
    print("  Nibble group (Z/16Z):")
    # Identity
    n0 = A("N0")
    for i in range(16):
        ni = nibble(i)
        assert dot_ext(n0, ni) == ni, f"N0 · N{i:X} ≠ N{i:X}"
        assert dot_ext(ni, n0) == ni, f"N{i:X} · N0 ≠ N{i:X}"
    print("    ✓ N0 is the identity")

    # Closure and commutativity
    for i in range(16):
        for j in range(16):
            ni, nj = nibble(i), nibble(j)
            result = dot_ext(ni, nj)
            expected = nibble((i + j) % 16)
            assert result == expected, f"N{i:X} · N{j:X} = {result}, expected {expected}"
    print("    ✓ Closed under addition mod 16")

    # Successor via i
    for x in range(16):
        nx = nibble(x)
        result = dot_ext(A("i"), nx)
        expected = nibble((x + 1) % 16)
        assert result == expected, f"i · N{x:X} = {result}, expected {expected}"
    print("    ✓ dot(i, N_x) = N_{(x+1) mod 16}")


def verify_self_id():
    """Verify all 34 new atoms satisfy dot(x, k) = x."""
    print("  Self-identification on k:")
    k = A("k")
    for atom in NEW_ATOMS:
        result = dot_ext(atom, k)
        assert result == atom, f"dot({atom}, k) = {result}, expected {atom}"
    print(f"    ✓ All {len(NEW_ATOMS)} new atoms satisfy dot(x, k) = x")


def verify_byte_arithmetic():
    """Verify byte arithmetic operations produce correct results."""
    print("  Byte arithmetic:")

    # Build byte values via B_LIT
    def bv(n):
        hi, lo = n >> 4, n & 0xF
        return dot_ext(dot_ext(A("B_LIT"), nibble(hi)), nibble(lo))

    # Verify all 256 byte values constructible
    for n in range(256):
        v = bv(n)
        assert isinstance(v, ByteVal), f"B_LIT failed for {n}: got {v}"
        assert v.val == n, f"B_LIT({n}) = bv({v.val})"
    print("    ✓ All 256 byte values constructible via B_LIT")

    # Test binary ops
    test_cases = [
        ("B_ADD", 100, 55, ByteVal(155)),
        ("B_ADD", 200, 100, ByteVal(44)),  # overflow
        ("B_SUB", 100, 55, ByteVal(45)),
        ("B_SUB", 10, 20, ByteVal(246)),   # underflow wraps
        ("B_MUL", 7, 8, ByteVal(56)),
        ("B_MUL", 16, 16, ByteVal(0)),     # overflow
        ("B_DIV", 100, 7, ByteVal(14)),
        ("B_DIV", 10, 0, A("p")),          # div by zero
        ("B_MOD", 100, 7, ByteVal(2)),
        ("B_MOD", 10, 0, A("p")),          # mod by zero
        ("B_AND", 0xAA, 0x0F, ByteVal(0x0A)),
        ("B_OR",  0xAA, 0x0F, ByteVal(0xAF)),
        ("B_XOR", 0xAA, 0x0F, ByteVal(0xA5)),
        ("B_SHL", 1, 7, ByteVal(128)),
        ("B_SHL", 0xFF, 4, ByteVal(0xF0)),
        ("B_SHR", 128, 7, ByteVal(1)),
        ("B_SHR", 0xFF, 4, ByteVal(0x0F)),
    ]
    for op, a, b, expected in test_cases:
        result = dot_ext(dot_ext(A(op), bv(a)), bv(b))
        assert result == expected, f"{op}({a}, {b}) = {result}, expected {expected}"
    print("    ✓ Binary arithmetic ops correct")

    # Test unary ops
    assert dot_ext(A("B_NOT"), bv(0x00)) == ByteVal(0xFF)
    assert dot_ext(A("B_NOT"), bv(0xFF)) == ByteVal(0x00)
    assert dot_ext(A("B_NOT"), bv(0xAA)) == ByteVal(0x55)
    print("    ✓ B_NOT correct")

    # Test predicates
    assert dot_ext(A("B_ZERO"), bv(0)) == A("⊤")
    assert dot_ext(A("B_ZERO"), bv(1)) == A("⊥")
    assert dot_ext(A("B_ZERO"), bv(255)) == A("⊥")
    print("    ✓ B_ZERO correct")

    assert dot_ext(dot_ext(A("B_EQ"), bv(42)), bv(42)) == A("⊤")
    assert dot_ext(dot_ext(A("B_EQ"), bv(42)), bv(43)) == A("⊥")
    print("    ✓ B_EQ correct")

    assert dot_ext(dot_ext(A("B_LT"), bv(3)), bv(4)) == A("⊤")
    assert dot_ext(dot_ext(A("B_LT"), bv(4)), bv(4)) == A("⊥")
    assert dot_ext(dot_ext(A("B_LT"), bv(5)), bv(4)) == A("⊥")
    print("    ✓ B_LT correct")

    # Test carry ops
    TOP, BOT = A("⊤"), A("⊥")
    assert dot_ext(dot_ext(dot_ext(A("B_ADC"), bv(100)), bv(50)), BOT) == ByteVal(150)
    assert dot_ext(dot_ext(dot_ext(A("B_ADC"), bv(100)), bv(50)), TOP) == ByteVal(151)
    assert dot_ext(dot_ext(dot_ext(A("B_ADC"), bv(200)), bv(100)), BOT) == ByteVal(44)
    print("    ✓ B_ADC correct")

    assert dot_ext(dot_ext(dot_ext(A("B_SBC"), bv(100)), bv(50)), BOT) == ByteVal(50)
    assert dot_ext(dot_ext(dot_ext(A("B_SBC"), bv(100)), bv(50)), TOP) == ByteVal(49)
    print("    ✓ B_SBC correct")

    # B_ROL: rotate left
    assert dot_ext(dot_ext(dot_ext(A("B_ROL"), bv(0x80)), bv(1)), BOT) == ByteVal(0x00)
    assert dot_ext(dot_ext(dot_ext(A("B_ROL"), bv(0x80)), bv(1)), TOP) == ByteVal(0x01)
    assert dot_ext(dot_ext(dot_ext(A("B_ROL"), bv(0x01)), bv(1)), BOT) == ByteVal(0x02)
    print("    ✓ B_ROL correct")


def verify_d2_preservation():
    """Verify Δ₂ fragment is preserved exactly."""
    print("  Δ₂ preservation:")

    # Import the original dot_iota for comparison
    from delta2_true_blackbox import dot_iota as dot_orig, ALL_ATOMS as ORIG_ATOMS

    # Check all Δ₁+Δ₂ atom pairs
    d2_names = NAMES_D1 + NAMES_D2
    for xn in d2_names:
        for yn in d2_names:
            orig = dot_orig(A(xn), A(yn))
            ext = dot_ext(A(xn), A(yn))
            assert orig == ext, f"dot({xn}, {yn}): orig={orig}, ext={ext}"
    print(f"    ✓ All {len(d2_names)}² atom-atom pairs match original Δ₂")

    # QUOTE/EVAL roundtrip
    for an in d2_names:
        q = dot_ext(A("QUOTE"), A(an))
        e = dot_ext(A("EVAL"), q)
        assert e == A(an), f"EVAL(QUOTE({an})) = {e}"
    print("    ✓ QUOTE/EVAL roundtrip preserved")

    # APP/UNAPP roundtrip
    for fn in ["e_D", "e_M", "i"]:
        for xn in ["i", "k", "a"]:
            node = dot_ext(dot_ext(A("APP"), A(fn)), A(xn))
            bundle = dot_ext(A("UNAPP"), node)
            left = dot_ext(bundle, A("⊤"))
            right = dot_ext(bundle, A("⊥"))
            assert left == A(fn), f"UNAPP(APP({fn},{xn}))·⊤ = {left}"
            assert right == A(xn), f"UNAPP(APP({fn},{xn}))·⊥ = {right}"
    print("    ✓ APP/UNAPP roundtrip preserved")


def main():
    print("=" * 60)
    print("  Δ₂+Byte VERIFICATION SUITE")
    print("=" * 60)
    print(f"\n  Total atoms: {len(ALL_ATOMS)}")
    print(f"    Δ₁ core: {len(NAMES_D1)}")
    print(f"    Δ₂ extensions: {len(NAMES_D2)}")
    print(f"    Byte ops: {len(NAMES_BYTE_OPS)}")
    print(f"    Nibbles: {len(NAMES_NIBBLES)}")
    print()

    verify_nibble_group()
    verify_self_id()
    verify_byte_arithmetic()
    verify_d2_preservation()

    print("\n" + "=" * 60)
    print("  All verifications passed. ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
