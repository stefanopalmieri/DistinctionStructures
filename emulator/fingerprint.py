"""
Structural fingerprints for the Kamea machine.

Each atom gets a canonical fingerprint ordinal (7 bits, 0x00-0x41) that
describes WHAT the atom IS, not WHERE it IS in a specific ROM permutation.

Programs store fingerprints in atom words. The Cayley ROM is addressed and
valued entirely in fingerprint space, so no runtime translation is needed.

Fingerprint encoding:
    0x00-0x0F: N0-NF         (nibbles, consecutive for range checks)
    0x10: ⊤                   0x11: ⊥
    0x12: i                   0x13: k
    0x14: a                   0x15: b
    0x16: e_I                 0x17: d_K
    0x18: m_I                 0x19: m_K
    0x1A: e_D                 0x1B: e_M
    0x1C: d_I                 0x1D: e_Σ
    0x1E: s_C                 0x1F: e_Δ
    0x20: p
    0x21: ALU_LOGIC           0x22: ALU_ARITH          0x23: ALU_ARITHC
    0x24: ALU_ZERO            0x25: ALU_COUT
    0x26: N_SUCC              0x27: QUALE
    0x28: QUOTE               0x29: EVAL
    0x2A: APP                 0x2B: UNAPP
    0x2C: IO_PUT              0x2D: IO_GET
    0x2E: IO_RDY              0x2F: IO_SEQ
    0x30-0x3F: W_PACK8..W_ROTR (16 W32 atoms)
    0x40: MUL16               0x41: MAC16
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical fingerprint ordinals (compile-time constants)
# ---------------------------------------------------------------------------

# Nibbles: 0x00-0x0F (value == fingerprint)
FP_N0  = 0x00; FP_N1  = 0x01; FP_N2  = 0x02; FP_N3  = 0x03
FP_N4  = 0x04; FP_N5  = 0x05; FP_N6  = 0x06; FP_N7  = 0x07
FP_N8  = 0x08; FP_N9  = 0x09; FP_NA  = 0x0A; FP_NB  = 0x0B
FP_NC  = 0x0C; FP_ND  = 0x0D; FP_NE  = 0x0E; FP_NF  = 0x0F

# D1 core
FP_TOP = 0x10; FP_BOT = 0x11
FP_I   = 0x12; FP_K   = 0x13
FP_A   = 0x14; FP_B   = 0x15
FP_E_I = 0x16; FP_D_K = 0x17
FP_M_I = 0x18; FP_M_K = 0x19
FP_E_D = 0x1A; FP_E_M = 0x1B
FP_D_I = 0x1C; FP_E_S = 0x1D
FP_S_C = 0x1E; FP_E_DELTA = 0x1F

# Extras
FP_P         = 0x20
FP_ALU_LOGIC = 0x21; FP_ALU_ARITH = 0x22; FP_ALU_ARITHC = 0x23
FP_ALU_ZERO  = 0x24; FP_ALU_COUT  = 0x25
FP_N_SUCC    = 0x26; FP_QUALE     = 0x27

# D2 (opaque atoms with structural targets)
FP_QUOTE  = 0x28; FP_EVAL  = 0x29
FP_APP    = 0x2A; FP_UNAPP = 0x2B
FP_IO_PUT = 0x2C; FP_IO_GET = 0x2D
FP_IO_RDY = 0x2E; FP_IO_SEQ = 0x2F

# W32 atoms
FP_W_PACK8 = 0x30; FP_W_LO   = 0x31; FP_W_HI   = 0x32; FP_W_MERGE = 0x33
FP_W_NIB   = 0x34; FP_W_ADD  = 0x35; FP_W_SUB  = 0x36; FP_W_CMP   = 0x37
FP_W_XOR   = 0x38; FP_W_AND  = 0x39; FP_W_OR   = 0x3A; FP_W_NOT   = 0x3B
FP_W_SHL   = 0x3C; FP_W_SHR  = 0x3D; FP_W_ROTL = 0x3E; FP_W_ROTR  = 0x3F

# MUL atoms
FP_MUL16 = 0x40; FP_MAC16 = 0x41

NUM_FP = 0x42  # total fingerprints

# ---------------------------------------------------------------------------
# Name ↔ fingerprint mappings
# ---------------------------------------------------------------------------

NAME_TO_FP: dict[str, int] = {
    # Nibbles
    "N0": FP_N0, "N1": FP_N1, "N2": FP_N2, "N3": FP_N3,
    "N4": FP_N4, "N5": FP_N5, "N6": FP_N6, "N7": FP_N7,
    "N8": FP_N8, "N9": FP_N9, "NA": FP_NA, "NB": FP_NB,
    "NC": FP_NC, "ND": FP_ND, "NE": FP_NE, "NF": FP_NF,
    # D1
    "⊤": FP_TOP, "⊥": FP_BOT,
    "i": FP_I, "k": FP_K,
    "a": FP_A, "b": FP_B,
    "e_I": FP_E_I, "d_K": FP_D_K,
    "m_I": FP_M_I, "m_K": FP_M_K,
    "e_D": FP_E_D, "e_M": FP_E_M,
    "d_I": FP_D_I, "e_Σ": FP_E_S,
    "s_C": FP_S_C, "e_Δ": FP_E_DELTA,
    "p": FP_P,
    # ALU/misc
    "ALU_LOGIC": FP_ALU_LOGIC, "ALU_ARITH": FP_ALU_ARITH,
    "ALU_ARITHC": FP_ALU_ARITHC,
    "ALU_ZERO": FP_ALU_ZERO, "ALU_COUT": FP_ALU_COUT,
    "N_SUCC": FP_N_SUCC, "QUALE": FP_QUALE,
    # D2
    "QUOTE": FP_QUOTE, "EVAL": FP_EVAL,
    "APP": FP_APP, "UNAPP": FP_UNAPP,
    # IO
    "IO_PUT": FP_IO_PUT, "IO_GET": FP_IO_GET,
    "IO_RDY": FP_IO_RDY, "IO_SEQ": FP_IO_SEQ,
    # W32
    "W_PACK8": FP_W_PACK8, "W_LO": FP_W_LO, "W_HI": FP_W_HI,
    "W_MERGE": FP_W_MERGE, "W_NIB": FP_W_NIB,
    "W_ADD": FP_W_ADD, "W_SUB": FP_W_SUB, "W_CMP": FP_W_CMP,
    "W_XOR": FP_W_XOR, "W_AND": FP_W_AND, "W_OR": FP_W_OR,
    "W_NOT": FP_W_NOT,
    "W_SHL": FP_W_SHL, "W_SHR": FP_W_SHR,
    "W_ROTL": FP_W_ROTL, "W_ROTR": FP_W_ROTR,
    # MUL
    "MUL16": FP_MUL16, "MAC16": FP_MAC16,
}

FP_TO_NAME: dict[int, str] = {v: k for k, v in NAME_TO_FP.items()}

# Precomputed sets for dispatch
FP_ALU_DISPATCH_SET = frozenset({FP_ALU_LOGIC, FP_ALU_ARITH, FP_ALU_ARITHC})

FP_W32_BINARY_OPS: dict[int, int] = {
    FP_W_ADD: 0, FP_W_SUB: 1, FP_W_CMP: 2,
    FP_W_XOR: 3, FP_W_AND: 4, FP_W_OR: 5,
    FP_W_SHL: 6, FP_W_SHR: 7, FP_W_ROTL: 8, FP_W_ROTR: 9,
}
