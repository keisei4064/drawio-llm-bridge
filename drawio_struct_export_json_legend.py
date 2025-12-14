#!/usr/bin/env python3
from __future__ import annotations

"""
drawio_struct_export_json_legend.py

Export draw.io (*.drawio.svg or mxfile XML) into an LLM-friendly JSON graph.

Key behaviors:
- Nodes: {"<id>": {"kind": "<kind>", "state": "<todo|wip|done>"?}, ...}
  - node id is the hierarchical path of labels joined by "/"
  - state is parsed from a marker emoji prefix in the node label (e.g., 🟦Controller)
  - marker prefix is stripped from the semantic id and displayed name
- Edges: [src, rel, dst, inferred]
  - inferred=False if:
      (a) edge has an explicit label, OR
      (b) edge style matches a relation learned from the legend Edge section
  - inferred=True only when falling back to heuristics.
- Legend learning:
  - Node section learns kind by mapping (kind label -> style signature)
  - Edge section learns rel by mapping (rel label -> edge style signature)
    - supports "label on the edge" OR "label as separate nearby text cell"
  - Marker section learns emoji -> {todo,wip,done} from a 2-column table
- Port resolution:
  - if draw.io creates port cells, edge source/target may point to the port id
  - this tool resolves endpoints by walking up parents to a labeled/container vertex

Usage:
  python drawio_struct_export_json_legend.py input.drawio.svg -o out.json [--debug]
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

# ---------------- vocab / mapping ----------------

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
    "ros2ノード": "ros2_node",
    "ROS2ノード": "ros2_node",
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
STATE_WORDS = ("todo", "wip", "done")

DEFAULT_MARKER_TO_STATE: dict[str, str] = {
    "🟦": "todo",
    "🪛": "wip",
    "✅": "done",
}

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

# ---------------- data model ----------------

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


# ---------------- xml helpers ----------------

def _strip_html(s: str) -> str:
    s2 = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    s2 = re.sub(r"<[^>]+>", "", s2)
    return html.unescape(s2).strip()

def _norm_space(s: str) -> str:
    s2 = s.replace("\u00A0", " ")
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
    processed: set[str] = set()

    def consume_cell(id_: Optional[str], label: str, style: str, parent: Optional[str], geom_elem: Optional[ET.Element], vertex_flag: str, edge_flag: str, src: Optional[str], dst: Optional[str]) -> None:
        if id_ is None or id_ in processed:
            return
        processed.add(id_)
        value = normalize_label(label)
        if vertex_flag == "1":
            st = parse_style(style)
            s_lower = style.lower()
            is_container = ("swimlane" in s_lower) or (st.get("container") == "1")
            if geom_elem is None:
                return
            x = float(geom_elem.get("x", "0"))
            y = float(geom_elem.get("y", "0"))
            w = float(geom_elem.get("width", "0"))
            h = float(geom_elem.get("height", "0"))
            geom = Geometry(x=x, y=y, w=w, h=h)
            vertices.append(Vertex(raw_id=id_, label=value, style=style, parent=parent, is_container=is_container, geom=geom))
        elif edge_flag == "1":
            if not src or not dst:
                return
            edges.append(Edge(raw_id=id_, src=src, dst=dst, label=value, style=style))

    for elem in root.iter():
        # Support <object label=...><mxCell .../></object> wrappers
        if elem.tag.endswith("object"):
            inner = next((c for c in elem if c.tag.endswith("mxCell")), None)
            if inner is None:
                continue
            consume_cell(
                id_=elem.get("id") or inner.get("id"),
                label=elem.get("label", inner.get("value", "")),
                style=inner.get("style", ""),
                parent=inner.get("parent"),
                geom_elem=inner.find("mxGeometry"),
                vertex_flag=inner.get("vertex", "0"),
                edge_flag=inner.get("edge", "0"),
                src=inner.get("source"),
                dst=inner.get("target"),
            )
            continue

        if not elem.tag.endswith("mxCell"):
            continue
        consume_cell(
            id_=elem.get("id"),
            label=elem.get("value", ""),
            style=elem.get("style", ""),
            parent=elem.get("parent"),
            geom_elem=elem.find("mxGeometry"),
            vertex_flag=elem.get("vertex", "0"),
            edge_flag=elem.get("edge", "0"),
            src=elem.get("source"),
            dst=elem.get("target"),
        )

    return Graph(vertices=vertices, edges=edges)

# ---------------- graph utilities ----------------

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

# ---------------- endpoint / port resolution ----------------

def is_probable_port(v: Vertex) -> bool:
    if v.is_container:
        return False
    if v.label.strip():
        return False
    st = parse_style(v.style)
    if st.get("portConstraint") is not None:
        return True
    if st.get("port") == "1":
        return True
    if v.geom.w <= 12.0 and v.geom.h <= 12.0:
        return True
    return False

def resolve_endpoint(raw_id: str, by_id: dict[str, Vertex]) -> Optional[str]:
    cur = by_id.get(raw_id)
    seen: set[str] = set()
    while cur is not None:
        if cur.raw_id in seen:
            break
        seen.add(cur.raw_id)
        if cur.is_container or cur.label.strip():
            return cur.raw_id
        if not is_probable_port(cur):
            return cur.raw_id
        if cur.parent is None or cur.parent in {"0", "1"}:
            break
        cur = by_id.get(cur.parent)
    return None

# ---------------- legend discovery & learning ----------------

def find_legend_root(by_id: dict[str, Vertex], children: dict[str, list[str]]) -> Optional[str]:
    for rid, v in by_id.items():
        if normalize_key(v.label) in {s.lower() for s in LEGEND_ROOT_LABELS}:
            return rid
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

def find_section_root(legend_root: str, section: str, by_id: dict[str, Vertex], children: dict[str, list[str]]) -> Optional[str]:
    targets = {s.lower() for s in LEGEND_SECTION_LABELS[section]}
    for sid in collect_subtree(legend_root, children):
        v = by_id.get(sid)
        if v is None:
            continue
        if normalize_key(v.label) in targets:
            return sid
    return None

def learn_node_kind_map(sample_ids: Iterable[str], by_id: dict[str, Vertex]) -> tuple[dict[str, str], list[str]]:
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

def _edge_midpoint(e: Edge, by_id: dict[str, Vertex]) -> Optional[tuple[float, float]]:
    src = resolve_endpoint(e.src, by_id)
    dst = resolve_endpoint(e.dst, by_id)
    if src is None or dst is None:
        return None
    vs = by_id.get(src)
    vt = by_id.get(dst)
    if vs is None or vt is None:
        return None
    return ((vs.geom.cx + vt.geom.cx) / 2.0, (vs.geom.cy + vt.geom.cy) / 2.0)

def learn_edge_rel_map_with_text(
    edge_root: str,
    g: Graph,
    by_id: dict[str, Vertex],
    children: dict[str, list[str]],
) -> tuple[dict[str, str], list[str]]:
    warnings: list[str] = []
    out: dict[str, str] = {}
    edge_sub = collect_subtree(edge_root, children)

    legend_edges: list[Edge] = []
    for e in g.edges:
        rs = resolve_endpoint(e.src, by_id)
        rt = resolve_endpoint(e.dst, by_id)
        if rs is None or rt is None:
            continue
        if rs in edge_sub and rt in edge_sub:
            legend_edges.append(e)

    label_vertices: list[Vertex] = []
    for vid in edge_sub:
        v = by_id.get(vid)
        if v is None:
            continue
        if REL_LABEL_TO_REL.get(normalize_key(v.label)):
            label_vertices.append(v)

    def nearest_label_for_edge(e: Edge) -> Optional[str]:
        mid = _edge_midpoint(e, by_id)
        if mid is None:
            return None
        mx, my = mid
        best: tuple[float, str] | None = None
        for lv in label_vertices:
            dy = abs(lv.geom.cy - my)
            dx = abs(lv.geom.cx - mx)
            score = dy * 3.0 + dx * 0.5
            if best is None or score < best[0]:
                best = (score, lv.label)
        if best is None or best[0] > 120.0:
            return None
        return best[1]

    learned = 0
    for e in legend_edges:
        explicit_label = bool(e.label.strip())
        rel_label = e.label.strip() or (nearest_label_for_edge(e) or "")
        rel = REL_LABEL_TO_REL.get(normalize_key(rel_label))
        if not rel:
            continue
        sig = signature_from_style(e.style, EDGE_SIG_KEYS)
        if not sig:
            # if label is explicitly set, treat as default style without warning
            if not explicit_label:
                warnings.append(f"legend-edge-style-empty: {rel_label} (raw_id={e.raw_id})")
            continue
        out[signature_key(sig)] = rel
        learned += 1

    if learned == 0:
        warnings.append("legend-edge-map-empty: no edge styles learned")
    return out, warnings

def learn_marker_map_from_table(marker_root: str, by_id: dict[str, Vertex], children: dict[str, list[str]]) -> tuple[dict[str, str], list[str]]:
    warnings: list[str] = []
    out: dict[str, str] = {}
    sub = collect_subtree(marker_root, children)
    candidates = [by_id[sid] for sid in sub if sid in by_id]

    by_parent: dict[str, list[Vertex]] = {}
    for v in candidates:
        if v.parent is None:
            continue
        by_parent.setdefault(v.parent, []).append(v)

    for _, vs in by_parent.items():
        # rows may be just [icon, label], so allow 2+
        if len(vs) < 2:
            continue
        for state in STATE_WORDS:
            tgt = [v for v in vs if normalize_key(v.label) == state]
            for state_cell in tgt:
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
                if best_icon is None or not best_icon.label:
                    continue
                marker = best_icon.label.lstrip()[:1].replace("\ufe0f", "")
                if marker:
                    out[marker] = state

    if not out:
        warnings.append("legend-marker-map-empty: no markers learned (will fall back to defaults)")
    return out, warnings

# ---------------- classification ----------------

def kind_from_style(style: str, learned_node_map: dict[str, str]) -> str:
    for key, kind in learned_node_map.items():
        if style_matches(required_from_sig_key(key), style):
            return kind
    st = parse_style(style)
    s_lower = style.lower()
    shape = (st.get("shape") or "").lower()

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
    if edge.label.strip():
        rel = REL_LABEL_TO_REL.get(normalize_key(edge.label), edge.label.strip())
        return rel, False
    for key, rel in learned_edge_map.items():
        if style_matches(required_from_sig_key(key), edge.style):
            return rel, False
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
    if not inferred:
        return rel, inferred
    if rel != "depends":
        return rel, inferred
    if dst_kind == "persistent_data" and src_kind in {"free_function", "ros2_node", "class"}:
        return "writes", True
    if src_kind == "persistent_data" and dst_kind in {"free_function", "ros2_node", "class"}:
        return "reads", True
    if dst_kind == "free_function" and src_kind in {"free_function", "ros2_node", "class"}:
        return "calls", True
    return rel, inferred

# ---------------- semantic IDs / marker parsing ----------------

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
        if base_label.strip():
            segs.append(sanitize_segment(base_label))
        if cur.parent is None or cur.parent in {"0", "1"}:
            break
        cur = by_id.get(cur.parent)
    if not segs:
        return raw_id
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

# ---------------- main ----------------

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

    legend_root = find_legend_root(by_id, children)
    legend_ids: set[str] = set()
    if legend_root is not None:
        legend_ids = collect_subtree(legend_root, children)

    node_root = find_section_root(legend_root, "node", by_id, children) if legend_root else None
    edge_root = find_section_root(legend_root, "edge", by_id, children) if legend_root else None
    marker_root = find_section_root(legend_root, "marker", by_id, children) if legend_root else None

    warnings: list[str] = []

    node_samples: set[str] = set()
    if node_root is not None:
        node_samples = collect_subtree(node_root, children)
    elif legend_root is not None:
        node_samples = legend_ids
    learned_node_map, w = learn_node_kind_map(node_samples, by_id)
    warnings.extend(w)

    learned_edge_map: dict[str, str] = {}
    if edge_root is not None:
        learned_edge_map, w2 = learn_edge_rel_map_with_text(edge_root, g, by_id, children)
        warnings.extend(w2)
    else:
        warnings.append("legend-edge-section-not-found: will rely on heuristics")

    marker_to_state: dict[str, str] = dict(DEFAULT_MARKER_TO_STATE)
    if marker_root is not None:
        learned_marker_map, w3 = learn_marker_map_from_table(marker_root, by_id, children)
        warnings.extend(w3)
        marker_to_state.update(learned_marker_map)
    else:
        warnings.append("legend-marker-section-not-found: will rely on defaults")

    sem_id = stable_semantic_ids(g.vertices, by_id, marker_to_state)

    resolved_edges: list[Edge] = []
    for e in g.edges:
        rs = resolve_endpoint(e.src, by_id)
        rt = resolve_endpoint(e.dst, by_id)
        if rs is None or rt is None:
            continue
        resolved_edges.append(Edge(raw_id=e.raw_id, src=rs, dst=rt, label=e.label, style=e.style))

    deg = compute_degree(resolved_edges)

    def is_legend_like_label(v: Vertex) -> bool:
        if LEGEND_LABEL_TO_KIND.get(v.label) or LEGEND_LABEL_TO_KIND.get(normalize_key(v.label)):
            return True
        if REL_LABEL_TO_REL.get(normalize_key(v.label)):
            return True
        if normalize_key(v.label) in STATE_WORDS:
            return True
        return False

    def should_drop_vertex(v: Vertex) -> bool:
        v_state = parse_state_from_label(v.label, marker_to_state)
        if (not args.include_legend) and (v.raw_id in legend_ids):
            return True
        if is_probable_port(v):
            return True
        if (not args.keep_isolated) and (deg.get(v.raw_id, 0) == 0) and (not v.is_container) and (v_state is None):
            return True
        if deg.get(v.raw_id, 0) == 0 and is_legend_like_label(v) and (v_state is None):
            return True
        if not v.label.strip() and not v.is_container:
            return True
        return False

    raw_kind: dict[str, str] = {v.raw_id: kind_from_style(v.style, learned_node_map) for v in g.vertices}

    raw_to_sem: dict[str, str] = {}
    nodes_out: dict[str, dict[str, str]] = {}

    for v in g.vertices:
        if should_drop_vertex(v):
            continue
        sid = sem_id[v.raw_id]
        raw_to_sem[v.raw_id] = sid
        obj: dict[str, str] = {"kind": raw_kind.get(v.raw_id, "class")}
        st = parse_state_from_label(v.label, marker_to_state)
        if st is not None:
            obj["state"] = st
        nodes_out[sid] = obj

    edges_out: list[list[Any]] = []
    for e in resolved_edges:
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
            "nodes map to objects with at least {'kind': ...} and optional {'state': ...}",
            "edges are [src, rel, dst, inferred] where inferred is true iff the relation was inferred (no explicit label and no legend-style match)",
        ],
        "nodes": dict(sorted(nodes_out.items(), key=lambda kv: kv[0])),
        "edges": edges_out,
        "warnings": warnings,
    }

    if args.debug:
        export["debug"] = {
            "legend_root_raw_id": legend_root,
            "node_section_raw_id": node_root,
            "edge_section_raw_id": edge_root,
            "marker_section_raw_id": marker_root,
            "learned_node_style_signatures": learned_node_map,
            "learned_edge_style_signatures": learned_edge_map,
            "marker_to_state": marker_to_state,
        }

    Path(args.output).write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
