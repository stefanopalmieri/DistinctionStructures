#!/usr/bin/env python3
"""
Visualize the Distinction Structures communication protocol signal.

Based on Gemini's concrete Layer B schema using Radio FSK + TDM.

Produces a multi-panel PNG:
  1. Layer A — Cayley table as color-coded heatmap (the self-interpreting payload)
  2. Frequency allocation map — 17 syntactic channels + 2 probe channels
  3. Layer B — Full TDM timeline (Phases 2–4)
  4. Epistemic bridge — the verification logic

Usage:
  python visualize_signal.py
  python visualize_signal.py --output signal.png --dpi 250
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import argparse


# ============================================================================
# Atom operation table
# ============================================================================

ATOMS = ['⊤', '⊥', 'i', 'k', 'a', 'b', 'e_I', 'e_D', 'e_M', 'e_Σ', 'e_Δ',
         'd_I', 'd_K', 'm_I', 'm_K', 's_C', 'p']

def dot(x, y):
    if x == '⊤': return '⊤'
    if x == '⊥': return '⊥'
    if x == 'e_I': return '⊤' if y in ('i', 'k') else '⊥'
    if x == 'd_K': return '⊤' if y in ('a', 'b') else '⊥'
    if x == 'm_K': return '⊤' if y == 'a' else '⊥'
    if x == 'm_I': return '⊥' if y == 'p' else '⊤'
    if x == 'e_D' and y == 'i': return 'd_I'
    if x == 'e_D' and y == 'k': return 'd_K'
    if x == 'e_M' and y == 'i': return 'm_I'
    if x == 'e_M' and y == 'k': return 'm_K'
    if x == 'e_Σ' and y == 's_C': return 'e_Δ'
    if x == 'e_Δ' and y == 'e_D': return 'd_I'
    if x == 'p' and y == '⊤': return '⊤'
    if y == '⊤' and x in ('i', 'k', 'a', 'b', 'd_I', 's_C'): return x
    return 'p'


# ============================================================================
# Color system
# ============================================================================

ATOM_COLOR_MAP = {
    '⊤': '#F0B429', '⊥': '#DE911D',
    'i': '#4C9AFF', 'k': '#2D7DD2',
    'a': '#36D7B7', 'b': '#1FAB89',
    'e_I': '#BB6BD9', 'd_K': '#A24DBF', 'm_K': '#8E44AD', 'm_I': '#7B2D8E',
    'e_D': '#FF6B6B', 'e_M': '#EE5A5A', 'e_Σ': '#D94848', 'e_Δ': '#C53636',
    'd_I': '#7E8CA0', 's_C': '#5BA4CF', 'p': '#8B9DAF',
}

BG        = '#0B0E14'
BG_PANEL  = '#111822'
BG_DEMO   = '#0D1520'
BG_ANNO   = '#141E2B'
TEXT       = '#D4DCE8'
TEXT_DIM   = '#7A8899'
BORDER     = '#2A3545'
ACCENT     = '#4C9AFF'
GOLD       = '#F0B429'
GREEN      = '#00E676'
RED        = '#FF5252'

BASE_FREQ = 1420.0
SPACING = 0.5
PROBE_F1 = BASE_FREQ + 20 * SPACING
PROBE_F2 = BASE_FREQ + 22 * SPACING


def hex_to_rgb(h):
    return [int(h[i:i+2], 16) / 255 for i in (1, 3, 5)]

def text_color_for_bg(hex_bg):
    r, g, b = hex_to_rgb(hex_bg)
    return '#111111' if (0.299*r + 0.587*g + 0.114*b) > 0.55 else '#FFFFFF'


# ============================================================================
# Panel 1: Cayley Table
# ============================================================================

def draw_cayley_table(ax):
    n = len(ATOMS)
    img = np.zeros((n, n, 3))
    for i, x in enumerate(ATOMS):
        for j, y in enumerate(ATOMS):
            img[i, j] = hex_to_rgb(ATOM_COLOR_MAP[dot(x, y)])

    ax.imshow(img, aspect='equal', interpolation='nearest')
    ax.set_xticks(range(n))
    ax.set_xticklabels(ATOMS, fontsize=5, rotation=60, ha='right', color=TEXT,
                       fontfamily='monospace')
    ax.set_yticks(range(n))
    ax.set_yticklabels(ATOMS, fontsize=5, color=TEXT, fontfamily='monospace')
    ax.set_xlabel('y  (right argument)', fontsize=7, color=TEXT_DIM, labelpad=6)
    ax.set_ylabel('x  (left operator)', fontsize=7, color=TEXT_DIM, labelpad=6)

    for i in range(n + 1):
        ax.axhline(i - 0.5, color=BG, linewidth=0.3)
        ax.axvline(i - 0.5, color=BG, linewidth=0.3)

    ax.tick_params(colors=TEXT_DIM, length=0)
    for sp in ax.spines.values():
        sp.set_color(BORDER); sp.set_linewidth(0.5)

    ax.set_title('LAYER A    Cayley Table    x · y → color-coded result',
                 fontsize=9, fontweight='bold', color=TEXT, pad=10,
                 fontfamily='monospace')
    ax.text(n + 0.3, n / 2, '289 entries\n17 × 17\n\nNo interpretation\nprovided.\n\nRecipient probes\nand recovers\nfull ontology.',
            fontsize=5, va='center', ha='left', color=TEXT_DIM,
            fontfamily='monospace', linespacing=1.6)


# ============================================================================
# Panel 2: Frequency Allocation
# ============================================================================

def draw_freq_map(ax):
    ax.set_facecolor(BG_PANEL)
    ax.set_xlim(-1, 28)
    ax.set_ylim(-3.2, 3.2)

    ax.set_title('FREQUENCY ALLOCATION    17 syntactic channels  +  2 probe channels',
                 fontsize=9, fontweight='bold', color=TEXT, pad=10,
                 fontfamily='monospace')

    for i, atom in enumerate(ATOMS):
        x = i * 1.3 + 0.2
        c = ATOM_COLOR_MAP[atom]
        tc = text_color_for_bg(c)
        rect = FancyBboxPatch((x, -0.5), 1.05, 1.8, boxstyle="round,pad=0.06",
                              facecolor=c, edgecolor='none', alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + 0.525, 0.55, atom, fontsize=4.5, ha='center', va='center',
                color=tc, fontweight='bold', fontfamily='monospace')
        ax.text(x + 0.525, -0.1, f'ν{i}', fontsize=3.5, ha='center', va='center',
                color=tc, alpha=0.6, fontfamily='monospace')

    bx0, bx1 = 0.2, 0.2 + 16 * 1.3 + 1.05
    ax.annotate('', xy=(bx1, -0.9), xytext=(bx0, -0.9),
                arrowprops=dict(arrowstyle='<->', color=ACCENT, lw=0.8))
    ax.text((bx0 + bx1) / 2, -1.25,
            f'Δ₁ Syntactic Band    ν₀–ν₁₆    ({BASE_FREQ:.0f}–{BASE_FREQ + 16*SPACING:.1f} MHz)',
            fontsize=5.5, ha='center', color=ACCENT, fontfamily='monospace')

    gx = bx1 + 0.8
    ax.text(gx + 0.3, 0.4, '···', fontsize=12, ha='center', va='center', color=TEXT_DIM)

    probes = [('f₁', GREEN, PROBE_F1, gx + 1.2), ('f₂', RED, PROBE_F2, gx + 2.7)]
    for label, c, freq, px in probes:
        rect = FancyBboxPatch((px, -0.5), 1.2, 1.8, boxstyle="round,pad=0.06",
                              facecolor=c, edgecolor='white', linewidth=0.6, alpha=0.9)
        ax.add_patch(rect)
        tc = text_color_for_bg(c)
        ax.text(px + 0.6, 0.55, label, fontsize=7, ha='center', va='center',
                color=tc, fontweight='bold', fontfamily='monospace')
        ax.text(px + 0.6, -0.1, f'{freq:.0f}', fontsize=4.5, ha='center', va='center',
                color=tc, alpha=0.7, fontfamily='monospace')

    pb0, pb1 = probes[0][3], probes[1][3] + 1.2
    ax.annotate('', xy=(pb1, -0.9), xytext=(pb0, -0.9),
                arrowprops=dict(arrowstyle='<->', color='#FF8A80', lw=0.8))
    ax.text((pb0 + pb1) / 2, -1.25, 'Probe Band (outside Δ₁ alphabet)',
            fontsize=5.5, ha='center', color='#FF8A80', fontfamily='monospace')

    roles = [
        ('Booleans ⊤ ⊥', '#F0B429'), ('Context tokens i k', '#4C9AFF'),
        ('κ-elements a b', '#36D7B7'), ('Testers e_I d_K m_K m_I', '#BB6BD9'),
        ('Encoders e_D e_M e_Σ e_Δ', '#FF6B6B'), ('Codes d_I  Token s_C', '#7E8CA0'),
        ('Default p', '#8B9DAF'),
    ]
    for j, (label, c) in enumerate(roles):
        col, row = j // 4, j % 4
        lx, ly = 1.0 + col * 11, -1.9 - row * 0.42
        rect = FancyBboxPatch((lx, ly - 0.1), 0.5, 0.28, boxstyle="round,pad=0.02",
                              facecolor=c, edgecolor='none', alpha=0.85)
        ax.add_patch(rect)
        ax.text(lx + 0.7, ly + 0.04, label, fontsize=4.5, va='center',
                color=TEXT_DIM, fontfamily='monospace')
    ax.axis('off')


# ============================================================================
# Panel 3: Layer B TDM Timeline
# ============================================================================

def draw_layer_b(ax):
    ax.set_facecolor(BG_PANEL)
    ax.set_xlim(-0.8, 34)
    ax.set_ylim(-2.5, 6.0)

    ax.set_title('LAYER B    Medium-Reflexive Grounding via Time-Division Multiplexing',
                 fontsize=9, fontweight='bold', color=TEXT, pad=10,
                 fontfamily='monospace')

    # Phase headers
    ax.text(5.5, 5.2, 'PHASE 2: Domain Anchoring (d_K)', fontsize=7.5, ha='center',
            color=ACCENT, fontweight='bold', fontfamily='monospace')
    ax.text(5.5, 4.65, '"These probe frequencies belong to a set"', fontsize=5,
            ha='center', color=TEXT_DIM, style='italic')

    ax.text(20.0, 5.2, 'PHASE 3: Actuality Grounding (m_K)', fontsize=7.5, ha='center',
            color=GOLD, fontweight='bold', fontfamily='monospace')
    ax.text(20.0, 4.65, '"This one is observed; that one is not"', fontsize=5,
            ha='center', color=TEXT_DIM, style='italic')

    ax.text(31.2, 5.2, 'PHASE 4', fontsize=7.5, ha='center',
            color=GREEN, fontweight='bold', fontfamily='monospace')
    ax.text(31.2, 4.65, 'Grounded ✓', fontsize=5, ha='center', color=GREEN, style='italic')

    # Dividers
    ax.axvline(x=12.8, color=BORDER, linestyle=':', lw=0.5, ymin=0.08, ymax=0.82)
    ax.axvline(x=27.8, color=BORDER, linestyle=':', lw=0.5, ymin=0.08, ymax=0.82)

    def wave(ax, x0, w, yc, color, amp=0.55, freq=3.5, alpha=0.9):
        t = np.linspace(0, w, 300)
        ax.plot(x0 + t, yc + amp * np.sin(2*np.pi*freq*t/w*4), color=color,
                linewidth=1.0, alpha=alpha)

    def noise(ax, x0, w, yc):
        t = np.linspace(0, w, 200)
        n = np.random.RandomState(42).normal(0, 0.05, len(t))
        ax.plot(x0 + t, yc + n, color='#444444', linewidth=0.4, alpha=0.4)

    def demo_box(ax, x0, w, step, desc, freq_label, color, silence=False, high=False):
        rect = FancyBboxPatch((x0, 0.0), w, 3.0, boxstyle="round,pad=0.12",
                              facecolor=BG_DEMO, edgecolor=color, linewidth=1.5, alpha=0.95)
        ax.add_patch(rect)
        if silence:
            noise(ax, x0+0.15, w-0.3, 1.5)
            ax.text(x0+w/2, 1.5, '∅', fontsize=20, ha='center', va='center',
                    color='#555555', fontweight='bold')
        else:
            wave(ax, x0+0.15, w-0.3, 1.5, color, amp=0.75 if high else 0.45)
        ax.text(x0+w/2, 3.5, step, fontsize=6, ha='center', color=color,
                fontweight='bold', fontfamily='monospace')
        ax.text(x0+w/2, 3.9, desc, fontsize=4.5, ha='center', color=TEXT_DIM,
                fontfamily='monospace')
        ax.text(x0+w/2, -0.35, freq_label, fontsize=4.5, ha='center', color=color,
                fontfamily='monospace', alpha=0.8)

    def anno_box(ax, x0, w, step, expr, note, border_color):
        rect = FancyBboxPatch((x0, 0.0), w, 3.0, boxstyle="round,pad=0.12",
                              facecolor=BG_ANNO, edgecolor=border_color,
                              linewidth=1.0, linestyle='--', alpha=0.9)
        ax.add_patch(rect)
        ax.text(x0+w/2, 1.7, expr, fontsize=8, ha='center', va='center',
                color=TEXT, fontfamily='monospace', fontweight='bold')
        ax.text(x0+w/2, 0.7, note, fontsize=5, ha='center', va='center',
                color=TEXT_DIM, fontfamily='monospace', style='italic')
        ax.text(x0+w/2, 3.5, step, fontsize=6, ha='center', color=border_color,
                fontweight='bold', fontfamily='monospace')
        ax.text(x0+w/2, -0.35, 'Δ₁ alphabet', fontsize=4.5, ha='center',
                color=border_color, fontfamily='monospace', alpha=0.6)

    wd, wa, g = 2.6, 3.2, 0.25
    x = 0.0

    # Phase 2
    demo_box(ax, x, wd, 'Step 2.1', 'Transmit CW', f'f₁ = {PROBE_F1:.0f} MHz', GREEN)
    x += wd + g
    anno_box(ax, x, wa, 'Step 2.2', 'd_K · s_f₁ = ⊤', '"s_f₁ is in the domain"', ACCENT)
    x += wa + g
    demo_box(ax, x, wd, 'Step 2.3', 'Transmit CW', f'f₂ = {PROBE_F2:.0f} MHz', RED)
    x += wd + g
    anno_box(ax, x, wa, 'Step 2.4', 'd_K · s_f₂ = ⊤', '"s_f₂ is in the domain"', ACCENT)
    x += wa + g + 0.6

    # Phase 3
    demo_box(ax, x, wd, 'Step 3.1', 'HIGH amplitude', f'f₁ = {PROBE_F1:.0f} MHz', GREEN, high=True)
    x += wd + g
    anno_box(ax, x, wa, 'Step 3.2', 'm_K · s_f₁ = ⊤', '"s_f₁ is ACTUAL"', GOLD)
    x += wa + g
    demo_box(ax, x, wd, 'Step 3.3', 'SILENCE', 'f₂  — no signal', '#555555', silence=True)
    x += wd + g
    anno_box(ax, x, wa, 'Step 3.4', 'm_K · s_f₂ = ⊥', '"s_f₂ is NOT actual"', GOLD)
    x += wa + g + 0.6

    # Phase 4 checkmark
    rect = FancyBboxPatch((x, 0.0), 3.0, 3.0, boxstyle="round,pad=0.15",
                          facecolor='#081208', edgecolor=GREEN, linewidth=2.0, alpha=0.95)
    ax.add_patch(rect)
    ax.text(x+1.5, 2.0, '✓', fontsize=28, ha='center', va='center',
            color=GREEN, fontweight='bold')
    ax.text(x+1.5, 0.7, '⊤ ↔ observed\n⊥ ↔ absent', fontsize=5.5,
            ha='center', va='center', color=GREEN, fontfamily='monospace',
            linespacing=1.5)

    # Time arrow
    ax.annotate('', xy=(33.5, -1.4), xytext=(-0.5, -1.4),
                arrowprops=dict(arrowstyle='->', color=TEXT_DIM, lw=0.8))
    ax.text(16.5, -1.8, 'time  →', fontsize=6, ha='center', color=TEXT_DIM,
            fontfamily='monospace')
    ax.text(16.5, -2.2,
            'Demo windows (manipulate medium)    ↔    Anno windows (describe in Δ₁)',
            fontsize=4.5, ha='center', color=TEXT_DIM, fontfamily='monospace')

    ax.axis('off')


# ============================================================================
# Panel 4: Epistemic Bridge
# ============================================================================

def draw_epistemic_bridge(ax):
    ax.set_facecolor(BG_PANEL)
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.8, 9.0)

    ax.set_title('RECIPIENT VERIFICATION    The Epistemic Bridge',
                 fontsize=9, fontweight='bold', color=TEXT, pad=10,
                 fontfamily='monospace')

    boxes = [
        ('1. RECOVERED FROM LAYER A  (pure algebra, no physics)',
         ACCENT, 7.5, 2.0,
         [('⊤  and  ⊥', 'the only left-absorbing elements', TEXT),
          ('d_K', 'the 2-element domain tester   |Dec(d_K)| = 2', TEXT),
          ('m_K', 'the 1-element actuality tester  |Dec(m_K)| = 1', TEXT),
          ('', 'All recovered by probing the Cayley table — no physics required', TEXT_DIM)]),
        ('2. CROSS-CHECKED AGAINST LAYER B  (algebra meets physics)',
         GOLD, 5.0, 2.0,
         [('Sender: m_K · s_f₁ = ⊤', 'Antenna at f₁?  ENERGY DETECTED  ✓', GREEN),
          ('Sender: m_K · s_f₂ = ⊥', 'Antenna at f₂?  SILENCE  ✓', RED),
          ('', 'Formal claims match physical observations', TEXT_DIM)]),
        ('3. EPISTEMIC ACHIEVEMENT',
         GREEN, 2.5, 2.2,
         [('⊤', '↔  physically observed phenomenon', GOLD),
          ('⊥', '↔  physically absent phenomenon', '#FF8A80'),
          ('', '', TEXT),
          ('The algebra is grounded in physical reality.', '', TEXT),
          ('All subsequent annotations are machine-verifiable.', '', GREEN)]),
    ]

    for title, color, y0, h, lines in boxes:
        rect = FancyBboxPatch((1, y0), 28, h, boxstyle="round,pad=0.2",
                              facecolor=BG, edgecolor=color, linewidth=1.5, alpha=0.95)
        ax.add_patch(rect)
        ax.text(2, y0 + h - 0.3, title, fontsize=6.5, fontweight='bold',
                color=color, fontfamily='monospace')
        ly = y0 + h - 0.7
        for sym, desc, tc in lines:
            if sym and len(sym) < 12:
                ax.text(3, ly, sym, fontsize=6, color=GOLD if sym in ('⊤','⊥') else tc,
                        fontfamily='monospace', fontweight='bold')
                ax.text(8, ly, desc, fontsize=5.5, color=tc, fontfamily='monospace')
            elif sym:
                ax.text(3, ly, sym, fontsize=6, color=tc, fontfamily='monospace', fontweight='bold')
                ax.text(15, ly, desc, fontsize=5.5, color=tc, fontfamily='monospace')
            elif desc:
                ax.text(3, ly, desc, fontsize=5, color=tc, fontfamily='monospace', style='italic')
            ly -= 0.36

    # Arrows
    for ya in [5.05, 2.55]:
        ax.annotate('', xy=(15, ya - 0.08), xytext=(15, ya + 0.08),
                    arrowprops=dict(arrowstyle='->', color=TEXT_DIM, lw=1.2,
                                    connectionstyle='arc3'))

    ax.text(15, -0.4,
            'FORMAL  STRUCTURE    ⟷    PHYSICAL  REALITY',
            fontsize=9, fontweight='bold', ha='center', color=GOLD,
            fontfamily='monospace',
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)])
    ax.axis('off')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="protocol_signal_v2.png")
    parser.add_argument("--dpi", type=int, default=250)
    args = parser.parse_args()

    fig = plt.figure(figsize=(20, 28), facecolor=BG)
    gs = gridspec.GridSpec(4, 1, height_ratios=[1.0, 0.55, 0.9, 0.75],
                           hspace=0.25, left=0.04, right=0.96, top=0.955, bottom=0.02)

    ax1 = fig.add_subplot(gs[0]); ax1.set_facecolor(BG_PANEL); draw_cayley_table(ax1)
    ax2 = fig.add_subplot(gs[1]); draw_freq_map(ax2)
    ax3 = fig.add_subplot(gs[2]); draw_layer_b(ax3)
    ax4 = fig.add_subplot(gs[3]); draw_epistemic_bridge(ax4)

    fig.text(0.5, 0.98, 'DISTINCTION  STRUCTURES  COMMUNICATION  PROTOCOL',
             fontsize=14, fontweight='bold', ha='center', color=TEXT, fontfamily='monospace')
    fig.text(0.5, 0.966,
             'Layer A (Self-Interpreting Algebra)  →  Layer B (Medium-Reflexive Grounding)',
             fontsize=9, ha='center', color=TEXT_DIM, fontfamily='monospace')

    plt.savefig(args.output, dpi=args.dpi, facecolor=BG, bbox_inches='tight', pad_inches=0.4)
    print(f"Saved: {args.output}  ({args.dpi} dpi)")

if __name__ == "__main__":
    main()
