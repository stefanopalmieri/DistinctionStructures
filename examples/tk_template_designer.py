#!/usr/bin/env python3
"""
Small Tkinter editor for ASCII mascot templates with Cayley-mapped color preview.

Features:
- Draw/erase template cells with mouse drag.
- Swap tokens in the Cayley ordering used to color the template.
- Show indexed Cayley cells and the exact source index for each preview cell.
- Save/load multiple templates in a JSON library.
- Export Python snippets for manual paste into kaput_host_bootstrap_demo.py.

Run:
  uv run examples/tk_template_designer.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import Any

# Allow imports from repository root when launched via examples/ path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from delta2_true_blackbox import ALL_ATOMS, AppNode, Atom, Partial, Quote, UnappBundle, dot_iota


LIBRARY_VERSION = 1
DEFAULT_LIBRARY_PATH = Path(__file__).with_name("ascii_template_library.json")

DEFAULT_TEMPLATE_LINES = [
    "     ****",
    "     **  *",
    "    ****",
    "   ******",
    "    ****",
    "    ****",
    "    *****",
    "   ** * **",
    "   ******* **",
    "           **",
    "   ******* *",
    "   ******* *",
    "    *****  *",
    "    *****  *",
    "    *****",
    "    *****",
    "    *   *",
    "    *   *",
    "    *   *",
    "    *   *",
    " *  *   *",
    "  ***********",
    "  **********",
    "  **********",
    "  *********",
    "  *********",
    "  *********",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
]

D1_CANONICAL_ORDER = [
    "⊤",
    "⊥",
    "i",
    "k",
    "a",
    "b",
    "e_I",
    "e_D",
    "e_M",
    "e_Σ",
    "e_Δ",
    "d_I",
    "d_K",
    "m_I",
    "m_K",
    "s_C",
    "p",
]

CUSTOM_TEMPLATE_ORDER = [
    "⊤",
    "⊥",
    "i",
    "k",
    "a",
    "b",
    "e_I",
    "e_D",
    "e_M",
    "e_Σ",
    "e_Δ",
    "d_I",
    "m_I",
    "QUOTE",
    "d_K",
    "m_K",
    "s_C",
    "UNAPP",
    "p",
    "EVAL",
    "APP",
]

TOKEN_GLYPH_PALETTE = [
    "#2f65ff",
    "#00b6e7",
    "#2bcf6b",
    "#bf4fff",
    "#ffcc33",
    "#ff4f4f",
    "#76adff",
    "#68e0ff",
    "#7df28f",
    "#dc8bff",
    "#ffe56c",
    "#f5f5f5",
]

SEMANTIC_COLORS = {
    "⊤": "#f2c94c",
    "⊥": "#3b82f6",
    "i": "#78c2ff",
    "k": "#4ad9e6",
    "a": "#39c46a",
    "b": "#7add8f",
    "e_I": "#d87cff",
    "e_D": "#ffe27a",
    "e_M": "#f5f5f5",
    "e_Σ": "#ff5f5f",
    "e_Δ": "#ff9b6b",
    "d_I": "#f0f0f0",
    "d_K": "#e79cff",
    "m_I": "#7be9ff",
    "m_K": "#8c8c8c",
    "s_C": "#8cb1ff",
    "p": "#505050",
}

STRUCTURE_COLORS = {
    "quote": "#e67dff",
    "partial": "#ffe27a",
    "app": "#ff7f7f",
    "bundle": "#8ce7ff",
    "atom_unknown": "#d9d9d9",
    "fallback": "#7a7a7a",
}

EDITOR_BG_ON = "#f3f3f3"
EDITOR_BG_OFF = "#1f1f1f"
EDITOR_GRID = "#4c4c4c"
PREVIEW_BG = "#101010"

EDITOR_CELL = 20
PREVIEW_CELL = 16
CAYLEY_CELL = 18
TARGET_FILLED_CELLS = 153


@dataclass
class TemplateRecord:
    name: str
    lines: list[str]
    order: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemplateRecord":
        return cls(
            name=str(data.get("name", "unnamed")),
            lines=[str(x) for x in data.get("lines", [])],
            order=[str(x) for x in data.get("order", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lines": self.lines,
            "order": self.order,
        }


def all_atom_names() -> list[str]:
    return [a.name for a in ALL_ATOMS]


def default_order() -> list[str]:
    return normalize_order(list(CUSTOM_TEMPLATE_ORDER))


def normalize_order(order: list[str]) -> list[str]:
    known = all_atom_names()
    out: list[str] = []
    for name in order:
        if name in known and name not in out:
            out.append(name)
    for name in known:
        if name not in out:
            out.append(name)
    return out


def lines_to_grid(lines: list[str]) -> list[list[bool]]:
    if not lines:
        return [[False]]
    width = max(1, max(len(row) for row in lines))
    grid: list[list[bool]] = []
    for row in lines:
        padded = row.ljust(width)
        grid.append([ch == "*" for ch in padded])
    return grid


def grid_to_lines(grid: list[list[bool]]) -> list[str]:
    lines: list[str] = []
    for row in grid:
        raw = "".join("*" if cell else " " for cell in row).rstrip()
        lines.append(raw)
    # Keep at least one row for editing convenience.
    if not lines:
        return [""]
    return lines


def ensure_rect_grid(grid: list[list[bool]]) -> list[list[bool]]:
    if not grid:
        return [[False]]
    width = max(1, max(len(row) for row in grid))
    out: list[list[bool]] = []
    for row in grid:
        rr = list(row) + [False] * (width - len(row))
        out.append(rr[:width])
    return out


def build_cayley_matrix(order: list[str]) -> list[list[Any]]:
    atoms = [Atom(name) for name in order]
    matrix: list[list[Any]] = []
    for x in atoms:
        row = [dot_iota(x, y) for y in atoms]
        matrix.append(row)
    return matrix


def token_color(name: str) -> str:
    idx = sum(ord(ch) for ch in name) % len(TOKEN_GLYPH_PALETTE)
    return TOKEN_GLYPH_PALETTE[idx]


def value_to_color(v: Any) -> str:
    if isinstance(v, Atom):
        if v.name in SEMANTIC_COLORS:
            return SEMANTIC_COLORS[v.name]
        return token_color(v.name)
    if isinstance(v, Quote):
        return STRUCTURE_COLORS["quote"]
    if isinstance(v, Partial):
        return STRUCTURE_COLORS["partial"]
    if isinstance(v, AppNode):
        return STRUCTURE_COLORS["app"]
    if isinstance(v, UnappBundle):
        return STRUCTURE_COLORS["bundle"]
    return STRUCTURE_COLORS["fallback"]


def text_color_for_bg(hex_color: str) -> str:
    if len(hex_color) != 7 or not hex_color.startswith("#"):
        return "#ffffff"
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
    except ValueError:
        return "#ffffff"
    # Standard perceived luminance heuristic.
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#101010" if lum >= 150 else "#f8f8f8"


def is_blank_value(v: Any) -> bool:
    return isinstance(v, Atom) and v.name == "p"


def sample_nonblank_values(matrix: list[list[Any]], count: int) -> tuple[list[tuple[int, Any]], int]:
    flat: list[tuple[int, Any]] = []
    idx = 0
    for row in matrix:
        for v in row:
            flat.append((idx, v))
            idx += 1
    nonblank = [(i, v) for i, v in flat if not is_blank_value(v)]
    if not nonblank:
        nonblank = flat if flat else [(0, Atom("p"))]
    if count <= 0:
        return [], len(nonblank)
    if count <= len(nonblank):
        # Greedy consumption in row-major order.
        sampled = nonblank[:count]
        return sampled, len(nonblank)
    # If template requests more cells than available nonblank entries,
    # continue greedily by wrapping from the start.
    sampled = [nonblank[i % len(nonblank)] for i in range(count)]
    return sampled, len(nonblank)


def default_library() -> dict[str, Any]:
    tpl = TemplateRecord(
        name="nova_ghost",
        lines=DEFAULT_TEMPLATE_LINES,
        order=default_order(),
    )
    return {
        "version": LIBRARY_VERSION,
        "active": tpl.name,
        "templates": [tpl.to_dict()],
    }


class TemplateDesignerApp:
    def __init__(self, root: tk.Tk, library_path: Path):
        self.root = root
        self.library_path = library_path

        self.templates: dict[str, TemplateRecord] = {}
        self.template_names: list[str] = []
        self.current_name: str | None = None

        self.grid: list[list[bool]] = [[False]]
        self.order: list[str] = default_order()
        self.paint_value: bool | None = None
        self.nonblank_pool_size = 0

        self.template_name_var = tk.StringVar()
        self.width_var = tk.StringVar(value="20")
        self.height_var = tk.StringVar(value="12")
        self.info_var = tk.StringVar(value="")
        self.filled_status_var = tk.StringVar(value="")

        self.template_list: tk.Listbox | None = None
        self.order_list: tk.Listbox | None = None
        self.editor_canvas: tk.Canvas | None = None
        self.preview_canvas: tk.Canvas | None = None
        self.cayley_canvas: tk.Canvas | None = None
        self.export_text: tk.Text | None = None
        self.filled_status_label: tk.Label | None = None
        self.order_modal: tk.Toplevel | None = None
        self.export_modal: tk.Toplevel | None = None

        self._build_ui()
        self._load_library()

    def _build_ui(self) -> None:
        self.root.title("DS ASCII Template Designer")
        self.root.geometry("1420x860")
        self.root.minsize(1200, 760)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, padding=8)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(2, weight=1)

        ttk.Label(top, text=f"Library: {self.library_path}").grid(row=0, column=0, sticky="w")
        ttk.Button(top, text="Save Library", command=self.save_library).grid(row=0, column=1, padx=(8, 0))
        ttk.Label(top, textvariable=self.info_var).grid(row=0, column=2, sticky="e")

        body = ttk.Frame(self.root, padding=8)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0, minsize=260)
        body.columnconfigure(1, weight=1, minsize=860)
        body.rowconfigure(0, weight=1)

        self._build_templates_panel(body)
        self._build_canvas_panel(body)

    def _build_templates_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Templates", padding=8)
        panel.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)

        name_row = ttk.Frame(panel)
        name_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        name_row.columnconfigure(1, weight=1)
        ttk.Label(name_row, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(name_row, textvariable=self.template_name_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self.template_list = tk.Listbox(panel, exportselection=False, height=18)
        self.template_list.grid(row=1, column=0, sticky="nsew")
        self.template_list.bind("<<ListboxSelect>>", self._on_template_select)

        btns = ttk.Frame(panel)
        btns.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        for idx in range(2):
            btns.columnconfigure(idx, weight=1)
        ttk.Button(btns, text="New", command=self.new_template).grid(row=0, column=0, sticky="ew")
        ttk.Button(btns, text="Clone", command=self.clone_template).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ttk.Button(btns, text="Delete", command=self.delete_template).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(btns, text="Save Template", command=self.save_current_template).grid(
            row=1, column=1, sticky="ew", padx=(6, 0), pady=(6, 0)
        )

        size_row = ttk.LabelFrame(panel, text="Grid Size", padding=8)
        size_row.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        size_row.columnconfigure(1, weight=1)
        size_row.columnconfigure(3, weight=1)
        ttk.Label(size_row, text="W").grid(row=0, column=0, sticky="w")
        ttk.Entry(size_row, textvariable=self.width_var, width=6).grid(row=0, column=1, sticky="w", padx=(4, 8))
        ttk.Label(size_row, text="H").grid(row=0, column=2, sticky="w")
        ttk.Entry(size_row, textvariable=self.height_var, width=6).grid(row=0, column=3, sticky="w", padx=(4, 0))
        ttk.Button(size_row, text="Resize Grid", command=self.resize_grid).grid(
            row=1, column=0, columnspan=4, sticky="ew", pady=(8, 0)
        )

        tools_row = ttk.Frame(panel)
        tools_row.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        tools_row.columnconfigure(0, weight=1)
        tools_row.columnconfigure(1, weight=1)
        ttk.Button(tools_row, text="Ordering Swap...", command=self.open_order_modal).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(tools_row, text="Python Export...", command=self.open_export_modal).grid(
            row=0, column=1, sticky="ew", padx=(6, 0)
        )

        ttk.Label(
            panel,
            text="Left drag: paint/erase\nRight click: erase",
            foreground="#666",
            justify="left",
        ).grid(row=5, column=0, sticky="w", pady=(10, 0))

        self.filled_status_label = tk.Label(panel, textvariable=self.filled_status_var, anchor="w")
        self.filled_status_label.grid(row=6, column=0, sticky="ew", pady=(10, 0))

    def _build_canvas_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent)
        panel.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
        panel.columnconfigure(0, weight=1)
        panel.columnconfigure(1, weight=1)
        panel.rowconfigure(1, weight=1)
        panel.rowconfigure(3, weight=1)

        ttk.Label(panel, text="Template Editor").grid(row=0, column=0, sticky="w")
        ttk.Label(panel, text="Cayley-Mapped Color Preview").grid(row=0, column=1, sticky="w")
        self.editor_canvas = tk.Canvas(panel, bg="#151515", highlightthickness=0)
        self.editor_canvas.grid(row=1, column=0, sticky="nsew", pady=(4, 10), padx=(0, 6))
        self.editor_canvas.bind("<Button-1>", self._on_editor_press_left)
        self.editor_canvas.bind("<B1-Motion>", self._on_editor_drag)
        self.editor_canvas.bind("<ButtonRelease-1>", self._on_editor_release)
        self.editor_canvas.bind("<Button-3>", self._on_editor_press_right)
        self.editor_canvas.bind("<B3-Motion>", self._on_editor_drag)
        self.editor_canvas.bind("<ButtonRelease-3>", self._on_editor_release)

        self.preview_canvas = tk.Canvas(panel, bg=PREVIEW_BG, highlightthickness=0)
        self.preview_canvas.grid(row=1, column=1, sticky="nsew", pady=(4, 10), padx=(6, 0))

        ttk.Label(panel, text="Cayley Diagram (cell index shown)").grid(
            row=2, column=0, columnspan=2, sticky="w"
        )
        self.cayley_canvas = tk.Canvas(panel, bg=PREVIEW_BG, highlightthickness=0)
        self.cayley_canvas.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(4, 0))

    def open_order_modal(self) -> None:
        if self.order_modal is not None and self.order_modal.winfo_exists():
            self.order_modal.focus_set()
            return

        modal = tk.Toplevel(self.root)
        modal.title("Cayley Ordering Swap")
        modal.geometry("380x680")
        modal.minsize(340, 520)
        modal.transient(self.root)
        modal.grab_set()
        modal.columnconfigure(0, weight=1)
        modal.rowconfigure(1, weight=1)

        ttk.Label(modal, text="Select 2 rows and press Swap.").grid(
            row=0, column=0, sticky="w", padx=10, pady=(10, 4)
        )
        self.order_list = tk.Listbox(modal, selectmode=tk.EXTENDED, exportselection=False)
        self.order_list.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))

        order_btns = ttk.Frame(modal)
        order_btns.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        for i in range(3):
            order_btns.columnconfigure(i, weight=1)
        ttk.Button(order_btns, text="Swap", command=self.swap_selected_tokens).grid(row=0, column=0, sticky="ew")
        ttk.Button(order_btns, text="Up", command=self.move_token_up).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(order_btns, text="Down", command=self.move_token_down).grid(row=0, column=2, sticky="ew")
        ttk.Button(order_btns, text="Scramble", command=self.scramble_order).grid(
            row=1, column=0, sticky="ew", pady=(6, 0)
        )
        ttk.Button(order_btns, text="Reset Canonical", command=self.reset_order).grid(
            row=1, column=1, columnspan=2, sticky="ew", padx=(6, 0), pady=(6, 0)
        )
        ttk.Button(order_btns, text="Close", command=lambda: self._close_order_modal(modal)).grid(
            row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0)
        )

        modal.protocol("WM_DELETE_WINDOW", lambda: self._close_order_modal(modal))
        self.order_modal = modal
        self.refresh_order_list()

    def _close_order_modal(self, modal: tk.Toplevel) -> None:
        self.order_list = None
        self.order_modal = None
        try:
            modal.grab_release()
        except tk.TclError:
            pass
        modal.destroy()

    def open_export_modal(self) -> None:
        if self.export_modal is not None and self.export_modal.winfo_exists():
            self.export_modal.focus_set()
            return

        modal = tk.Toplevel(self.root)
        modal.title("Python Export")
        modal.geometry("560x620")
        modal.minsize(420, 360)
        modal.transient(self.root)
        modal.grab_set()
        modal.columnconfigure(0, weight=1)
        modal.rowconfigure(1, weight=1)

        ttk.Label(modal, text="Export snippet for manual paste").grid(
            row=0, column=0, sticky="w", padx=10, pady=(10, 4)
        )
        self.export_text = tk.Text(modal, wrap="none")
        self.export_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))

        btns = ttk.Frame(modal)
        btns.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        btns.columnconfigure(2, weight=1)
        ttk.Button(btns, text="Refresh", command=self.refresh_export).grid(row=0, column=0, sticky="ew")
        ttk.Button(btns, text="Copy", command=self.copy_export).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(btns, text="Close", command=lambda: self._close_export_modal(modal)).grid(
            row=0, column=2, sticky="ew"
        )

        modal.protocol("WM_DELETE_WINDOW", lambda: self._close_export_modal(modal))
        self.export_modal = modal
        self.refresh_export()

    def _close_export_modal(self, modal: tk.Toplevel) -> None:
        self.export_text = None
        self.export_modal = None
        try:
            modal.grab_release()
        except tk.TclError:
            pass
        modal.destroy()

    def _load_library(self) -> None:
        if not self.library_path.exists():
            data = default_library()
            self._write_library(data)
        else:
            try:
                data = json.loads(self.library_path.read_text(encoding="utf-8"))
            except Exception:
                data = default_library()

        raw_templates = data.get("templates", [])
        loaded: dict[str, TemplateRecord] = {}
        for item in raw_templates:
            rec = TemplateRecord.from_dict(item)
            if not rec.name:
                continue
            rec.lines = rec.lines or [""]
            rec.order = normalize_order(rec.order)
            loaded[rec.name] = rec
        if not loaded:
            d = default_library()["templates"][0]
            rec = TemplateRecord.from_dict(d)
            loaded[rec.name] = rec

        self.templates = loaded
        self.template_names = sorted(self.templates.keys())
        active = str(data.get("active", ""))
        if active not in self.templates:
            active = self.template_names[0]

        self._refresh_template_listbox()
        self._load_template(active)
        self.info_var.set(f"Loaded {len(self.templates)} template(s)")

    def _write_library(self, data: dict[str, Any]) -> None:
        self.library_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _refresh_template_listbox(self) -> None:
        assert self.template_list is not None
        self.template_list.delete(0, tk.END)
        for name in self.template_names:
            self.template_list.insert(tk.END, name)

    def _load_template(self, name: str) -> None:
        rec = self.templates[name]
        self.current_name = name
        self.template_name_var.set(name)
        self.grid = ensure_rect_grid(lines_to_grid(rec.lines))
        self.order = normalize_order(rec.order)
        self.width_var.set(str(len(self.grid[0])))
        self.height_var.set(str(len(self.grid)))
        self._refresh_template_selection(name)
        self.refresh_all()

    def _refresh_template_selection(self, name: str) -> None:
        assert self.template_list is not None
        idx = self.template_names.index(name)
        self.template_list.selection_clear(0, tk.END)
        self.template_list.selection_set(idx)
        self.template_list.activate(idx)
        self.template_list.see(idx)

    def _on_template_select(self, _event: Any) -> None:
        assert self.template_list is not None
        sel = self.template_list.curselection()
        if not sel:
            return
        name = self.template_names[sel[0]]
        if name == self.current_name:
            return
        self._store_current_in_memory()
        self._load_template(name)

    def _store_current_in_memory(self) -> None:
        if self.current_name is None:
            return
        new_name = self.template_name_var.get().strip() or self.current_name
        lines = grid_to_lines(self.grid)
        order = normalize_order(self.order)

        if new_name != self.current_name:
            if new_name in self.templates:
                messagebox.showerror("Name Exists", f"Template '{new_name}' already exists.")
                self.template_name_var.set(self.current_name)
                return
            rec = self.templates.pop(self.current_name)
            rec.name = new_name
            rec.lines = lines
            rec.order = order
            self.templates[new_name] = rec
            self.current_name = new_name
        else:
            rec = self.templates[self.current_name]
            rec.lines = lines
            rec.order = order

        self.template_names = sorted(self.templates.keys())
        self._refresh_template_listbox()
        self._refresh_template_selection(self.current_name)

    def save_current_template(self) -> None:
        self._store_current_in_memory()
        self.save_library()

    def save_library(self) -> None:
        self._store_current_in_memory()
        active = self.current_name if self.current_name else (self.template_names[0] if self.template_names else "")
        data = {
            "version": LIBRARY_VERSION,
            "active": active,
            "templates": [self.templates[name].to_dict() for name in sorted(self.templates.keys())],
        }
        self._write_library(data)
        self.info_var.set(f"Saved {len(self.templates)} template(s)")

    def on_close(self) -> None:
        try:
            self.save_library()
        finally:
            if self.order_modal is not None and self.order_modal.winfo_exists():
                self._close_order_modal(self.order_modal)
            if self.export_modal is not None and self.export_modal.winfo_exists():
                self._close_export_modal(self.export_modal)
            self.root.destroy()

    def new_template(self) -> None:
        name = simpledialog.askstring("New Template", "Template name:")
        if not name:
            return
        name = name.strip()
        if not name:
            return
        if name in self.templates:
            messagebox.showerror("Exists", f"Template '{name}' already exists.")
            return
        self._store_current_in_memory()
        w = max(1, int(self.width_var.get() or "20"))
        h = max(1, int(self.height_var.get() or "12"))
        lines = ["" for _ in range(h)]
        rec = TemplateRecord(name=name, lines=lines, order=default_order())
        self.templates[name] = rec
        self.template_names = sorted(self.templates.keys())
        self._refresh_template_listbox()
        self._load_template(name)
        self.resize_grid(force_width=w, force_height=h)

    def clone_template(self) -> None:
        if self.current_name is None:
            return
        clone_name = simpledialog.askstring("Clone Template", "New name for clone:")
        if not clone_name:
            return
        clone_name = clone_name.strip()
        if not clone_name:
            return
        if clone_name in self.templates:
            messagebox.showerror("Exists", f"Template '{clone_name}' already exists.")
            return
        self._store_current_in_memory()
        rec = self.templates[self.current_name]
        self.templates[clone_name] = TemplateRecord(
            name=clone_name,
            lines=list(rec.lines),
            order=list(rec.order),
        )
        self.template_names = sorted(self.templates.keys())
        self._refresh_template_listbox()
        self._load_template(clone_name)

    def delete_template(self) -> None:
        if self.current_name is None:
            return
        if len(self.templates) <= 1:
            messagebox.showerror("Cannot Delete", "At least one template must remain.")
            return
        if not messagebox.askyesno("Delete Template", f"Delete '{self.current_name}'?"):
            return
        gone = self.current_name
        self.templates.pop(gone, None)
        self.template_names = sorted(self.templates.keys())
        self._refresh_template_listbox()
        self._load_template(self.template_names[0])

    def resize_grid(self, force_width: int | None = None, force_height: int | None = None) -> None:
        try:
            new_w = force_width if force_width is not None else int(self.width_var.get())
            new_h = force_height if force_height is not None else int(self.height_var.get())
        except ValueError:
            messagebox.showerror("Invalid Size", "Width and height must be integers.")
            return
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        old = ensure_rect_grid(self.grid)
        old_h = len(old)
        old_w = len(old[0])

        resized = [[False for _ in range(new_w)] for _ in range(new_h)]
        for r in range(min(old_h, new_h)):
            for c in range(min(old_w, new_w)):
                resized[r][c] = old[r][c]

        self.grid = resized
        self.width_var.set(str(new_w))
        self.height_var.set(str(new_h))
        self.refresh_all()

    def _event_to_cell(self, event: tk.Event, cell_size: int) -> tuple[int, int] | None:
        if not self.grid:
            return None
        h = len(self.grid)
        w = len(self.grid[0])
        c = event.x // cell_size
        r = event.y // cell_size
        if r < 0 or c < 0 or r >= h or c >= w:
            return None
        return r, c

    def _on_editor_press_left(self, event: tk.Event) -> None:
        rc = self._event_to_cell(event, EDITOR_CELL)
        if rc is None:
            return
        r, c = rc
        self.paint_value = not self.grid[r][c]
        self._set_cell(r, c, self.paint_value)

    def _on_editor_press_right(self, event: tk.Event) -> None:
        rc = self._event_to_cell(event, EDITOR_CELL)
        if rc is None:
            return
        r, c = rc
        self.paint_value = False
        self._set_cell(r, c, False)

    def _on_editor_drag(self, event: tk.Event) -> None:
        if self.paint_value is None:
            return
        rc = self._event_to_cell(event, EDITOR_CELL)
        if rc is None:
            return
        r, c = rc
        self._set_cell(r, c, self.paint_value)

    def _on_editor_release(self, _event: tk.Event) -> None:
        self.paint_value = None

    def _set_cell(self, r: int, c: int, value: bool) -> None:
        if self.grid[r][c] == value:
            return
        self.grid[r][c] = value
        self.refresh_all()

    def selected_order_indices(self) -> list[int]:
        if self.order_list is None:
            return []
        return [int(i) for i in self.order_list.curselection()]

    def swap_selected_tokens(self) -> None:
        idx = self.selected_order_indices()
        if len(idx) != 2:
            messagebox.showinfo("Swap", "Select exactly two tokens to swap.")
            return
        a, b = idx
        self.order[a], self.order[b] = self.order[b], self.order[a]
        self.refresh_order_list(selected=[a, b])
        self.refresh_preview_and_export()

    def move_token_up(self) -> None:
        idx = self.selected_order_indices()
        if len(idx) != 1:
            return
        i = idx[0]
        if i <= 0:
            return
        self.order[i - 1], self.order[i] = self.order[i], self.order[i - 1]
        self.refresh_order_list(selected=[i - 1])
        self.refresh_preview_and_export()

    def move_token_down(self) -> None:
        idx = self.selected_order_indices()
        if len(idx) != 1:
            return
        i = idx[0]
        if i >= len(self.order) - 1:
            return
        self.order[i + 1], self.order[i] = self.order[i], self.order[i + 1]
        self.refresh_order_list(selected=[i + 1])
        self.refresh_preview_and_export()

    def scramble_order(self) -> None:
        random.shuffle(self.order)
        self.refresh_order_list()
        self.refresh_preview_and_export()

    def reset_order(self) -> None:
        self.order = default_order()
        self.refresh_order_list()
        self.refresh_preview_and_export()

    def refresh_all(self) -> None:
        self.grid = ensure_rect_grid(self.grid)
        self.width_var.set(str(len(self.grid[0])))
        self.height_var.set(str(len(self.grid)))
        self.refresh_editor()
        self.refresh_order_list()
        self.refresh_preview_and_export()

    def refresh_editor(self) -> None:
        assert self.editor_canvas is not None
        canvas = self.editor_canvas
        canvas.delete("all")

        h = len(self.grid)
        w = len(self.grid[0]) if h else 1
        canvas.config(width=w * EDITOR_CELL, height=h * EDITOR_CELL)
        for r in range(h):
            for c in range(w):
                x0 = c * EDITOR_CELL
                y0 = r * EDITOR_CELL
                x1 = x0 + EDITOR_CELL
                y1 = y0 + EDITOR_CELL
                color = EDITOR_BG_ON if self.grid[r][c] else EDITOR_BG_OFF
                canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=EDITOR_GRID, width=1)

    def refresh_order_list(self, selected: list[int] | None = None) -> None:
        self.order = normalize_order(self.order)
        if self.order_list is None:
            return
        lb = self.order_list
        lb.delete(0, tk.END)
        for i, name in enumerate(self.order):
            lb.insert(tk.END, f"{i:02d}  {name}")
        lb.selection_clear(0, tk.END)
        if selected:
            for i in selected:
                if 0 <= i < len(self.order):
                    lb.selection_set(i)
            lb.see(selected[0])

    def refresh_preview_and_export(self) -> None:
        self.refresh_preview()
        self.refresh_export()

    def refresh_preview(self) -> None:
        assert self.preview_canvas is not None
        canvas = self.preview_canvas
        canvas.delete("all")
        canvas.config(bg=PREVIEW_BG)

        h = len(self.grid)
        w = len(self.grid[0]) if h else 1
        canvas.config(width=w * PREVIEW_CELL, height=h * PREVIEW_CELL)

        matrix = build_cayley_matrix(self.order)
        filled_positions = [(r, c) for r, row in enumerate(self.grid) for c, v in enumerate(row) if v]
        sampled_entries, self.nonblank_pool_size = sample_nonblank_values(matrix, len(filled_positions))
        sampled_indices = {src_idx for src_idx, _ in sampled_entries}
        self.refresh_cayley(matrix, sampled_indices)

        for idx, (r, c) in enumerate(filled_positions):
            src_idx, src_value = sampled_entries[idx]
            color = value_to_color(src_value)
            x0 = c * PREVIEW_CELL
            y0 = r * PREVIEW_CELL
            x1 = x0 + PREVIEW_CELL
            y1 = y0 + PREVIEW_CELL
            canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=color)
            canvas.create_text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                text=str(src_idx),
                fill=text_color_for_bg(color),
                font=("TkDefaultFont", 7),
            )

        filled = len(filled_positions)
        self._update_filled_status(filled)
        self.info_var.set(
            f"template={self.current_name or '-'}  "
            f"size={w}x{h}  filled={filled}  "
            f"order={len(self.order)}  nonblank_pool={self.nonblank_pool_size}  "
            f"preview_indexed=yes"
        )

    def refresh_cayley(self, matrix: list[list[Any]], highlighted: set[int]) -> None:
        if self.cayley_canvas is None:
            return
        canvas = self.cayley_canvas
        canvas.delete("all")
        canvas.config(bg=PREVIEW_BG)
        n = len(matrix)
        if n == 0:
            return

        cell = CAYLEY_CELL
        canvas.config(width=n * cell, height=n * cell)
        for r, row in enumerate(matrix):
            for c, value in enumerate(row):
                idx = r * n + c
                color = value_to_color(value)
                x0 = c * cell
                y0 = r * cell
                x1 = x0 + cell
                y1 = y0 + cell
                border = "#f8f8f8" if idx in highlighted else "#1b1b1b"
                bw = 2 if idx in highlighted else 1
                canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=border, width=bw)
                canvas.create_text(
                    (x0 + x1) / 2,
                    (y0 + y1) / 2,
                    text=str(idx),
                    fill=text_color_for_bg(color),
                    font=("TkDefaultFont", 7),
                )

    def _update_filled_status(self, filled: int) -> None:
        if self.filled_status_label is None:
            return
        if filled == TARGET_FILLED_CELLS:
            self.filled_status_var.set(f"Filled Target: {filled}/{TARGET_FILLED_CELLS} OK")
            self.filled_status_label.config(fg="#2bcf6b")
            return
        if filled < TARGET_FILLED_CELLS:
            diff_text = f"need +{TARGET_FILLED_CELLS - filled}"
        else:
            diff_text = f"over by {filled - TARGET_FILLED_CELLS}"
        self.filled_status_var.set(
            f"Filled Target: {filled}/{TARGET_FILLED_CELLS} MISMATCH ({diff_text})"
        )
        self.filled_status_label.config(fg="#ff5f5f")

    def refresh_export(self) -> None:
        if self.export_text is None:
            return
        lines = grid_to_lines(self.grid)
        order = list(self.order)
        text = self._python_export_text(lines, order)
        self.export_text.delete("1.0", tk.END)
        self.export_text.insert("1.0", text)

    def _python_export_text(self, lines: list[str], order: list[str]) -> str:
        out: list[str] = []
        out.append("# Paste into examples/kaput_host_bootstrap_demo.py")
        out.append("")
        out.append("WIZARD_TEMPLATE = [")
        for row in lines:
            out.append(f"    {row!r},")
        out.append("]")
        out.append("")
        out.append("# Example ordering to use as your mapped Cayley order")
        out.append("CUSTOM_TEMPLATE_ORDER = [")
        for tok in order:
            out.append(f"    {tok!r},")
        out.append("]")
        out.append("")
        out.append(f"# filled cells: {sum(row.count('*') for row in lines)}")
        return "\n".join(out)

    def copy_export(self) -> None:
        if self.export_text is None:
            self.open_export_modal()
            return
        txt = self.export_text.get("1.0", tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(txt)
        self.info_var.set("Export copied to clipboard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tk ASCII template designer for DS Cayley-mapped mascot art.")
    parser.add_argument(
        "--library",
        type=Path,
        default=DEFAULT_LIBRARY_PATH,
        help=f"Path to template library JSON (default: {DEFAULT_LIBRARY_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    app = TemplateDesignerApp(root, args.library)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
