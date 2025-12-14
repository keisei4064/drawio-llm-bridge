#!/usr/bin/env python3
from __future__ import annotations

"""
drawio_struct_export_json_legend.py

Export a draw.io diagram into an LLM-friendly minimal JSON graph, while learning
node kinds / edge relations / state markers dynamically from the in-diagram legend.

Legend rules assumed (your "above rules"):
- Node kind learning:
  - Legend contains sample nodes whose labels are kind names (e.g., クラス/interface/package/module/...)
  - Script learns style signatures -> kind.
- Edge relation learning:
  - Legend contains sample edges whose labels are relation names (depends/inherits/composes/...)
  - Script learns edge style signatures -> rel.
  - For non-legend edges: if edge has explicit label, it is treated as ground-truth (inferred=False).
- Marker (state) learning:
  - Legend contains a marker table where each row has:
    [left icon cell with an emoji marker] [right cell text: todo|wip|done]
  - Script learns emoji marker -> state.
- Real nodes carry the marker as the FIRST character in the label (e.g., 🟦ClassName).
- Semantic node IDs are hierarchical paths joined by '/', and the marker prefix is stripped
  so IDs don't change when state changes.

Output format (minimal):
{
  "meta": {"nodes": N, "edges": M},
  "notes": [...],
  "nodes": { "<node_id>": "<kind>", ... },
  "state": { "<node_id>": "todo|wip|done", ... },   # omitted if empty
  "edges": [ ["src","rel","dst", inferred_bool], ... ],
  "warnings": [...]
}

Python: 3.10+
"""

import argparse
import base64
import html
import json
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

_CONTENT_RE = re.compile(r'content="([^"]+)"', re.DOTALL)

# ----------- Config / vocab -----------

KINDS: tuple[str, ...] = (
    "package",
    "sub_module",
    "interface",
    "class",
    "data_class",
    "free_function",
    "ros2_node",
    "persistent_data",
    "memo",
)

REL_NAMES: tuple[str, ...] = (
    "depends",
    "inherits",
    "implements",
    "composes",
    "aggregates",
    "calls",
    "reads",
    "writes",
)

LEGEND_LABEL_TO_KIND: dict[str, str] = {
    # English
    "package": "package",
    "sub-module": "sub_module",
    "sub_module": "sub_module",
    "sub module": "sub_module",
    "module": "sub_module",
    "interface": "interface",
    "class": "class",
    "data class": "data_class",
    "dataclass": "data_class",
    "free function": "free_function",
    "function": "free_function",
    "ros2 node": "ros2_node",
    "persistent data": "persistent_data",
    "memo": "memo",
    # Japanese
    "パッケージ": "package",
    "サブモジュール": "sub_module",
    "インターフェース": "interface",
    "クラス": "class",
    "データクラス": "data_class",
    "フリー関数": "free_function",
    "ROS2ノード": "ros2_node",
    "ros2ノード": "ros2_node",
    "永続化データ": "persistent_data",
    "メモ": "memo",
    "ノート": "memo",
}

REL_LABEL_TO_REL: dict[str, str] = {
    "depends": "depends",
    "dependency": "depends",
    "uses": "depends",
    "imports": "depends",
    "inherits": "inherits",
    "extends": "inherits",
    "implements": "implements",
    "composes": "composes",
    "composition": "composes",
    "aggregates": "aggregates",
    "aggregation": "aggregates",
    "calls": "calls",
    "reads": "reads",
    "writes": "writes",
}

LEGEND_ROOT_LABELS = {"legend", "凡例", "レジェンド"}
LEGEND_SECTION_LABELS = {
    "node": {"node", "ノード", "Node"},
    "edge": {"edge", "エッジ", "Edge"},
    "marker": {"marker", "マーカー", "Marker"},
}

# State names expected in marker legend
STATE_WORDS = ("todo", "wip", "done")

# Default marker map (fallback if marker legend can't be learned)
DEFAULT_MARKER_TO_STATE: dict[str, str] = {
    "🟦": "todo",
    "🪛": "wip",
    "🔧": "wip",
    "✅": "done",
}

# Style keys to include in signatures
NODE_SIG_KEYS: tuple[str, ...] = (
    "shape",
    "rounded",
    "container",
    "swimlane",
    "group",
    "backgroundOutline",
    "double",
)

EDGE_SIG_KEYS: tuple[str, ...] = (
    "startArrow",
    "endArrow",
    "startFill",
    "endFill",
    "dashed",
)

# ----------- Data model -----------

@dataclass(frozen=True, slots=True)
class Geometry:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass(frozen=True, slots=True)
class Vertex:
    raw_id: str
    label: str
    style: str
    parent: Optional[str]
    is_container: bool
    geom: Geometry


@dataclass(frozen=True, slots=True)
class Edge:
    raw_id: str
    src: str
    dst: str
    label: str
    style: str


@dataclass(frozen=True, slots=True)
class Graph:
    vertices: list[Vertex]
    edges: list[Edge]


# ----------- XML parsing -----------

def _strip_html(s: str) -> str:
    s2 = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    s2 = re.sub(r"<[^>]+>", "", s2)
    return html.unescape(s2).strip()

def _norm_space(s: str) -> str:
    s2 = s.replace("\u00A0", " ")  # NBSP
    s2 = s2.replace("\r", "\n")
    s2 = re.sub(r"\s*\n\s*", " ", s2)
    s2 = re.sub(r"\s+", " ", s2)
    return s2.strip()

def normalize_label(s: str) -> str:
    return _norm_space(_strip_html(s))

def normalize_key(s: str) -> str:
    return normalize_label(s).lower()

def parse_style(style: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in style.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            out[part] = "1"
    return out

def decode_diagram_text(diagram_text: str) -> str:
    s = diagram_text.strip()
    if "<mxGraphModel" in s:
        return s

    raw = base64.b64decode(s)
    inflated: Optional[bytes] = None
    for wbits in (-15, 15, 31):
        try:
            inflated = zlib.decompress(raw, wbits=wbits)
            break
        except Exception:
            continue
    if inflated is None:
        raise ValueError("Failed to decompress <diagram> payload.")

    decoded = urllib.parse.unquote(inflated.decode("utf-8", errors="strict"))
    if "<mxGraphModel" not in decoded:
        raise ValueError("Decoded payload did not contain <mxGraphModel>.")
    return decoded

def extract_mxfile_xml_from_svg(svg_text: str) -> str:
    m = _CONTENT_RE.search(svg_text)
    if m is None:
        raise ValueError('No content="...mxfile..." attribute found in SVG.')
    return html.unescape(m.group(1))

def extract_mxgraphmodel_from_file(path: Path, diagram_name: Optional[str]) -> str:
    text = path.read_text(encoding="utf-8")
    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".drawio.svg") or path.suffix.lower() == ".svg":
        mxfile_xml = extract_mxfile_xml_from_svg(text)
        root = ET.fromstring(mxfile_xml)
    else:
        root = ET.fromstring(text)

    if root.tag != "mxfile":
        raise ValueError(f"Expected <mxfile>, got <{root.tag}>")

    diagrams = list(root.findall("diagram"))
    if not diagrams:
        raise ValueError("No <diagram> found.")
    if diagram_name is None:
        chosen = diagrams[0]
    else:
        chosen = next((d for d in diagrams if d.get("name") == diagram_name), None)
        if chosen is None:
            avail = [d.get("name") for d in diagrams]
            raise ValueError(f'No diagram name="{diagram_name}". Available: {avail}')

    if chosen.text is None:
        raise ValueError("Chosen <diagram> has no payload.")
    return decode_diagram_text(chosen.text)

def parse_mxgraphmodel(mxgraph_xml: str) -> Graph:
    root = ET.fromstring(mxgraph_xml)
    vertices: list[Vertex] = []
    edges: list[Edge] = []

    for cell in root.iter():
        if not cell.tag.endswith("mxCell"):
            continue
        cid = cell.get("id")
        if cid is None:
            continue

        style = cell.get("style", "")
        parent = cell.get("parent")
        value = normalize_label(cell.get("value", ""))

        if cell.get("vertex") == "1":
            st = parse_style(style)
            s_lower = style.lower()
            is_container = ("swimlane" in s_lower) or (st.get("container") == "1")
            geom_elem = cell.find("mxGeometry")
            if geom_elem is None:
                continue
            x = float(geom_elem.get("x", "0"))
            y = float(geom_elem.get("y", "0"))
            w = float(geom_elem.get("width", "0"))
            h = float(geom_elem.get("height", "0"))
            geom = Geometry(x=x, y=y, w=w, h=h)
            # keep empty labels for containers (they may be used as legend boxes)
            if not value and not is_container:
                continue
            vertices.append(Vertex(raw_id=cid, label=value, style=style, parent=parent, is_container=is_container, geom=geom))

        elif cell.get("edge") == "1":
            src = cell.get("source")
            dst = cell.get("target")
            if not src or not dst:
                continue
            edges.append(Edge(raw_id=cid, src=src, dst=dst, label=value, style=style))

    return Graph(vertices=vertices, edges=edges)

# ----------- Graph utilities -----------

def build_vertex_index(vertices: Sequence[Vertex]) -> tuple[dict[str, Vertex], dict[str, list[str]]]:
    by_id: dict[str, Vertex] = {v.raw_id: v for v in vertices}
    children: dict[str, list[str]] = {}
    for v in vertices:
        if v.parent is None or v.parent in {"0", "1"}:
            continue
        children.setdefault(v.parent, []).append(v.raw_id)
    for pid in children:
        children[pid].sort(key=lambda s: int(s) if s.isdigit() else s)
    return by_id, children

def collect_subtree(root_id: str, children: dict[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    stack = [root_id]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for ch in children.get(cur, []):
            stack.append(ch)
    return seen

def compute_degree(edges: Sequence[Edge]) -> dict[str, int]:
    deg: dict[str, int] = {}
    for e in edges:
        deg[e.src] = deg.get(e.src, 0) + 1
        deg[e.dst] = deg.get(e.dst, 0) + 1
    return deg

def signature_from_style(style: str, keys: Sequence[str]) -> dict[str, str]:
    st = parse_style(style)
    sig: dict[str, str] = {}
    for k in keys:
        if k in st:
            sig[k] = st[k]
    return sig

def signature_key(sig: dict[str, str]) -> str:
    return ",".join(f"{k}={sig[k]}" for k in sorted(sig.keys()))

def style_matches(required: dict[str, str], style: str) -> bool:
    st = parse_style(style)
    for k, v in required.items():
        if st.get(k) != v:
            return False
    return True

def required_from_sig_key(key: str) -> dict[str, str]:
    req: dict[str, str] = {}
    for kv in key.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split("=", 1)
        req[k] = v
    return req

# ----------- Legend discovery & learning -----------

def find_legend_root(by_id: dict[str, Vertex], children: dict[str, list[str]]) -> Optional[str]:
    # 1) explicit label
    for rid, v in by_id.items():
        if normalize_key(v.label) in {s.lower() for s in LEGEND_ROOT_LABELS}:
            return rid

    # 2) best container with many kind-labeled descendants
    best: tuple[int, str] | None = None
    for rid, v in by_id.items():
        if not v.is_container:
            continue
        sub = collect_subtree(rid, children)
        hit = 0
        for sid in sub:
            vv = by_id.get(sid)
            if vv is None:
                continue
            if LEGEND_LABEL_TO_KIND.get(vv.label) or LEGEND_LABEL_TO_KIND.get(normalize_key(vv.label)):
                hit += 1
        if hit >= 4:
            cand = (hit, rid)
            if best is None or cand[0] > best[0]:
                best = cand
    return best[1] if best else None

def find_section_root(
    legend_root: str,
    section: str,
    by_id: dict[str, Vertex],
    children: dict[str, list[str]],
) -> Optional[str]:
    targets = {s.lower() for s in LEGEND_SECTION_LABELS[section]}
    for sid in collect_subtree(legend_root, children):
        v = by_id.get(sid)
        if v is None:
            continue
        if normalize_key(v.label) in targets:
            return sid
    return None

def learn_node_kind_map(
    sample_ids: Iterable[str],
    by_id: dict[str, Vertex],
) -> tuple[dict[str, str], list[str]]:
    warnings: list[str] = []
    out: dict[str, str] = {}
    for rid in sample_ids:
        v = by_id.get(rid)
        if v is None:
            continue
        k = LEGEND_LABEL_TO_KIND.get(v.label) or LEGEND_LABEL_TO_KIND.get(normalize_key(v.label))
        if not k:
            continue
        sig = signature_from_style(v.style, NODE_SIG_KEYS)
        if not sig:
            warnings.append(f"legend-node-style-empty: {v.label} (raw_id={v.raw_id})")
            continue
        out[signature_key(sig)] = k
    if not out:
        warnings.append("legend-node-map-empty: no node styles learned")
    return out, warnings

def learn_edge_rel_map(
    sample_edge_ids: Iterable[str],
    edge_by_id: dict[str, Edge],
) -> tuple[dict[str, str], list[str]]:
    warnings: list[str] = []
    out: dict[str, str] = {}
    for eid in sample_edge_ids:
        e = edge_by_id.get(eid)
        if e is None:
            continue
        rel = REL_LABEL_TO_REL.get(normalize_key(e.label))
        if not rel:
            continue
        sig = signature_from_style(e.style, EDGE_SIG_KEYS)
        if not sig:
            warnings.append(f"legend-edge-style-empty: {e.label} (raw_id={e.raw_id})")
            continue
        out[signature_key(sig)] = rel
    if not out:
        warnings.append("legend-edge-map-empty: no edge styles learned")
    return out, warnings

def learn_marker_map_from_table(
    marker_root: str,
    by_id: dict[str, Vertex],
    children: dict[str, list[str]],
) -> tuple[dict[str, str], list[str]]:
    """
    Try to learn marker->state from a two-column marker table:
      [icon cell with emoji marker] [text cell: todo|wip|done]
    Implementation:
      - Find all vertices inside marker_root whose label is one of STATE_WORDS.
      - For each, find the closest vertex to its left (same parent) with overlapping y-range.
      - If that vertex has a non-empty label, take its first char as marker (strip VS16).
    """
    warnings: list[str] = []
    out: dict[str, str] = {}
    sub = collect_subtree(marker_root, children)
    candidates = [by_id[sid] for sid in sub if sid in by_id]

    # Index by parent for "same row" heuristics
    by_parent: dict[str, list[Vertex]] = {}
    for v in candidates:
        if v.parent is None:
            continue
        by_parent.setdefault(v.parent, []).append(v)

    for parent_id, vs in by_parent.items():
        # only consider table-like groups (enough vertices)
        if len(vs) < 4:
            continue
        # find state label cells
        for state in STATE_WORDS:
            tgt = [v for v in vs if normalize_key(v.label) == state]
            for state_cell in tgt:
                # find left icons: overlapping vertical center and to the left
                best_icon: Vertex | None = None
                best_dx = 1e18
                for cand in vs:
                    if cand.raw_id == state_cell.raw_id:
                        continue
                    if cand.geom.cx >= state_cell.geom.cx:
                        continue
                    if abs(cand.geom.cy - state_cell.geom.cy) > max(6.0, min(cand.geom.h, state_cell.geom.h) * 0.8):
                        continue
                    dx = state_cell.geom.cx - cand.geom.cx
                    if dx < best_dx:
                        best_dx = dx
                        best_icon = cand
                if best_icon is None:
                    warnings.append(f"legend-marker-row-no-icon: state={state} raw_id={state_cell.raw_id}")
                    continue
                if not best_icon.label:
                    warnings.append(f"legend-marker-icon-empty: state={state} icon_raw_id={best_icon.raw_id}")
                    continue
                marker = best_icon.label.lstrip()[:1].replace("\ufe0f", "")
                if not marker:
                    warnings.append(f"legend-marker-icon-nochar: state={state} icon_raw_id={best_icon.raw_id}")
                    continue
                out[marker] = state
    if not out:
        warnings.append("legend-marker-map-empty: no markers learned (will fall back to defaults)")
    return out, warnings

# ----------- Classification / export -----------

def kind_from_style(style: str, learned_node_map: dict[str, str]) -> str:
    # exact learned match
    for key, kind in learned_node_map.items():
        if style_matches(required_from_sig_key(key), style):
            return kind

    st = parse_style(style)
    s_lower = style.lower()
    shape = (st.get("shape") or "").lower()

    # fallback heuristics (kept conservative)
    if st.get("container") == "1" and shape in {"umlframe"}:
        return "package"
    if st.get("container") == "1" and shape in {"folder"}:
        return "sub_module"
    if "swimlane" in s_lower and shape == "umlframe":
        return "package"
    if shape == "hexagon":
        return "interface"
    if shape in {"note", "note2"}:
        return "memo"
    if shape.startswith("cylinder"):
        return "persistent_data"
    if st.get("backgroundOutline") == "1" and shape == "process":
        return "ros2_node"
    if st.get("backgroundOutline") == "1" and shape == "cube":
        return "data_class"
    if st.get("rounded") == "1":
        return "free_function"
    return "class"

def infer_rel_from_edge(edge: Edge, learned_edge_map: dict[str, str]) -> tuple[str, bool]:
    # explicit label wins
    if edge.label.strip():
        rel = REL_LABEL_TO_REL.get(normalize_key(edge.label), edge.label.strip())
        return rel, False

    # learned style match
    for key, rel in learned_edge_map.items():
        if style_matches(required_from_sig_key(key), edge.style):
            return rel, True

    # fallback heuristics
    st = parse_style(edge.style)
    dashed = st.get("dashed") == "1"
    start_arrow = (st.get("startArrow") or "").lower()
    end_arrow = (st.get("endArrow") or "").lower()
    start_fill = st.get("startFill", "1")
    end_fill = st.get("endFill", "1")

    if "diamond" in start_arrow:
        return ("composes" if start_fill == "1" else "aggregates"), True

    triangle_like = end_arrow in {"block", "blockthin", "triangle", "open", "classic"}
    if triangle_like and end_fill == "0":
        return ("implements" if dashed else "inherits"), True

    return "depends", True

def rich_relation(rel: str, inferred: bool, src_kind: str, dst_kind: str) -> tuple[str, bool]:
    # only refine inferred depends
    if not inferred or rel != "depends":
        return rel, inferred
    if dst_kind == "persistent_data" and src_kind in {"free_function", "ros2_node", "class"}:
        return "writes", True
    if src_kind == "persistent_data" and dst_kind in {"free_function", "ros2_node", "class"}:
        return "reads", True
    if dst_kind == "free_function" and src_kind in {"free_function", "ros2_node", "class"}:
        return "calls", True
    return rel, inferred

def sanitize_segment(seg: str) -> str:
    s = seg.strip()
    if not s:
        return "_"
    s = s.replace("/", "／")
    s = re.sub(r"\s+", " ", s)
    return s

def strip_state_marker_prefix(label: str, marker_to_state: dict[str, str]) -> str:
    if not label:
        return label
    s = label.lstrip()
    if not s:
        return label
    ch = s[0].replace("\ufe0f", "")
    if ch in marker_to_state:
        return s[1:].lstrip()
    return label

def build_path_id(raw_id: str, by_id: dict[str, Vertex], marker_to_state: dict[str, str]) -> str:
    segs: list[str] = []
    cur = by_id.get(raw_id)
    seen: set[str] = set()
    while cur is not None:
        if cur.raw_id in seen:
            break
        seen.add(cur.raw_id)
        base_label = strip_state_marker_prefix(cur.label, marker_to_state)
        seg = sanitize_segment(base_label if base_label else cur.raw_id)
        segs.append(seg)
        if cur.parent is None or cur.parent in {"0", "1"}:
            break
        cur = by_id.get(cur.parent)
    return "/".join(reversed(segs))

def stable_semantic_ids(vertices: Sequence[Vertex], by_id: dict[str, Vertex], marker_to_state: dict[str, str]) -> dict[str, str]:
    bucket: dict[str, list[str]] = {}
    for v in vertices:
        base = build_path_id(v.raw_id, by_id, marker_to_state)
        bucket.setdefault(base, []).append(v.raw_id)

    out: dict[str, str] = {}
    for base, rids in bucket.items():
        rids_sorted = sorted(rids, key=lambda x: int(x) if x.isdigit() else x)
        if len(rids_sorted) == 1:
            out[rids_sorted[0]] = base
        else:
            for i, rid in enumerate(rids_sorted, start=1):
                out[rid] = f"{base}#{i}"
    return out

def parse_state_from_label(label: str, marker_to_state: dict[str, str]) -> Optional[str]:
    s = label.lstrip()
    if not s:
        return None
    ch = s[0].replace("\ufe0f", "")
    return marker_to_state.get(ch)

def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="drawio_struct_export_json_legend.py")
    ap.add_argument("input", help="Input: *.drawio.svg or mxfile XML")
    ap.add_argument("-o", "--output", required=True, help="Output JSON path")
    ap.add_argument("--diagram", default=None, help="Diagram name (if multiple)")
    ap.add_argument("--include-legend", action="store_true", help="Include legend nodes/edges in output")
    ap.add_argument("--keep-isolated", action="store_true", help="Keep degree-0 nodes (default: drop non-container isolated nodes)")
    ap.add_argument("--no-rich-relations", action="store_true", help="Disable calls/reads/writes refinement")
    ap.add_argument("--debug", action="store_true", help="Include debug section with learned maps")
    args = ap.parse_args(list(argv))

    mx = extract_mxgraphmodel_from_file(Path(args.input), diagram_name=args.diagram)
    g = parse_mxgraphmodel(mx)
    by_id, children = build_vertex_index(g.vertices)
    edge_by_id: dict[str, Edge] = {e.raw_id: e for e in g.edges}
    deg = compute_degree(g.edges)

    warnings: list[str] = []

    legend_root = find_legend_root(by_id, children)
    legend_ids: set[str] = set()
    if legend_root is not None:
        legend_ids = collect_subtree(legend_root, children)

    # Find sections inside legend (optional)
    node_root = find_section_root(legend_root, "node", by_id, children) if legend_root else None
    edge_root = find_section_root(legend_root, "edge", by_id, children) if legend_root else None
    marker_root = find_section_root(legend_root, "marker", by_id, children) if legend_root else None

    # --- Learn node kind map ---
    node_samples: set[str] = set()
    if node_root is not None:
        node_samples = collect_subtree(node_root, children)
    elif legend_root is not None:
        node_samples = legend_ids
    else:
        # legendless fallback: use isolated kind-labeled nodes
        node_samples = {v.raw_id for v in g.vertices if deg.get(v.raw_id, 0) == 0 and (LEGEND_LABEL_TO_KIND.get(v.label) or LEGEND_LABEL_TO_KIND.get(normalize_key(v.label)))}

    learned_node_map, w = learn_node_kind_map(node_samples, by_id)
    warnings.extend(w)

    # --- Learn edge rel map ---
    edge_samples: set[str] = set()
    if edge_root is not None and legend_root is not None:
        # pick edges whose endpoints are inside edge_root subtree
        edge_sub = collect_subtree(edge_root, children)
        for e in g.edges:
            if e.src in edge_sub and e.dst in edge_sub:
                edge_samples.add(e.raw_id)
    elif legend_root is not None:
        # use edges fully inside legend
        for e in g.edges:
            if e.src in legend_ids and e.dst in legend_ids:
                edge_samples.add(e.raw_id)
    else:
        # legendless fallback: any labeled edge that matches known rel names
        for e in g.edges:
            if REL_LABEL_TO_REL.get(normalize_key(e.label)):
                edge_samples.add(e.raw_id)

    learned_edge_map, w2 = learn_edge_rel_map(edge_samples, edge_by_id)
    warnings.extend(w2)

    # --- Learn marker map ---
    marker_to_state: dict[str, str] = dict(DEFAULT_MARKER_TO_STATE)
    if marker_root is not None:
        learned_marker_map, w3 = learn_marker_map_from_table(marker_root, by_id, children)
        warnings.extend(w3)
        marker_to_state.update(learned_marker_map)
    else:
        # try to learn from anywhere inside legend if exists
        if legend_root is not None:
            learned_marker_map, w3 = learn_marker_map_from_table(legend_root, by_id, children)
            warnings.extend(w3)
            marker_to_state.update(learned_marker_map)

    # Build stable semantic IDs (marker-stripped)
    sem_id = stable_semantic_ids(g.vertices, by_id, marker_to_state)

    # Classify kinds
    raw_kind: dict[str, str] = {v.raw_id: kind_from_style(v.style, learned_node_map) for v in g.vertices}

    def should_drop_vertex(v: Vertex) -> bool:
        if (not args.include_legend) and (v.raw_id in legend_ids):
            return True
        if (not args.keep_isolated) and (deg.get(v.raw_id, 0) == 0) and (not v.is_container):
            return True
        # Drop isolated legend-like kind labels even if legend detection failed
        if deg.get(v.raw_id, 0) == 0 and (LEGEND_LABEL_TO_KIND.get(v.label) or LEGEND_LABEL_TO_KIND.get(normalize_key(v.label))):
            return True
        # Drop marker table words if they ended up outside legend (failsafe)
        if normalize_key(v.label) in STATE_WORDS and deg.get(v.raw_id, 0) == 0:
            return True
        return False

    raw_to_sem: dict[str, str] = {}
    nodes_out: dict[str, str] = {}
    state_out: dict[str, str] = {}
    for v in g.vertices:
        if should_drop_vertex(v):
            continue
        sid = sem_id[v.raw_id]
        raw_to_sem[v.raw_id] = sid
        nodes_out[sid] = raw_kind.get(v.raw_id, "class")
        st = parse_state_from_label(v.label, marker_to_state)
        if st is not None:
            state_out[sid] = st

    edges_out: list[list[Any]] = []
    for e in g.edges:
        if e.src not in raw_to_sem or e.dst not in raw_to_sem:
            continue
        rel, inferred = infer_rel_from_edge(e, learned_edge_map)
        if not args.no_rich_relations:
            rel, inferred = rich_relation(rel, inferred, raw_kind.get(e.src, "class"), raw_kind.get(e.dst, "class"))
        edges_out.append([raw_to_sem[e.src], rel, raw_to_sem[e.dst], bool(inferred)])

    edges_out.sort(key=lambda t: (str(t[0]), str(t[1]), str(t[2]), bool(t[3])))

    export: dict[str, Any] = {
        "meta": {"nodes": len(nodes_out), "edges": len(edges_out)},
        "notes": [
            "node ids are hierarchical paths joined by '/'",
            "edges are [src, rel, dst, inferred] where inferred is true iff the relation was inferred (no explicit label)",
        ],
        "nodes": dict(sorted(nodes_out.items(), key=lambda kv: kv[0])),
        "edges": edges_out,
        "warnings": warnings,
    }
    if state_out:
        export["state"] = dict(sorted(state_out.items(), key=lambda kv: kv[0]))

    if args.debug:
        export["debug"] = {
            "legend_root_raw_id": legend_root,
            "learned_node_style_signatures": learned_node_map,
            "learned_edge_style_signatures": learned_edge_map,
            "marker_to_state": marker_to_state,
            "node_section_raw_id": node_root,
            "edge_section_raw_id": edge_root,
            "marker_section_raw_id": marker_root,
        }

    Path(args.output).write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
