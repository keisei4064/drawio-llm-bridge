#!/usr/bin/env python3
from __future__ import annotations

"""
drawio_struct_export_json_min.py (v3)

LLM-friendly *minimal* JSON export for draw.io diagrams.

Output:
{
  "meta": {"nodes": N, "edges": M},
  "notes": [
    "node ids are hierarchical paths joined by '/'",
    "edges are [src, rel, dst, inferred]"
  ],
  "nodes": { "<node_id>": "<kind>", ... },
  "edges": [
    ["<src_id>", "<rel>", "<dst_id>", true],
    ...
  ],
  "warnings": [...]
}

Improvements vs v2
- Smarter legend detection:
  - label matches (legend|凡例|レジェンド) OR
  - container with >=4 children whose labels map to known kinds
- Auto-drop "legend item" nodes even if legend root is not detected:
  - nodes whose label is a kind label (e.g., "クラス", "package", ...) AND have degree 0
- Unicode whitespace normalization (NBSP -> space)
- Optional auto-drop isolated nodes (degree 0) (default: ON for non-containers)
- Deterministic edge sorting

Usage:
  python drawio_struct_export_json_min.py input.drawio.svg -o graph.json

Options:
  --include-legend
  --exclude-kind KIND          (repeatable)
  --exclude-name-regex REGEX   (repeatable)
  --no-rich-relations          disable calls/reads/writes heuristic upgrades
  --keep-isolated              keep degree-0 nodes (default is to drop non-container isolated nodes)
  --debug                      include learned style map and filters hit counts
  --dump-style-map FILE.json
  --use-style-map FILE.json
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
from typing import Any, Optional, Sequence, Tuple

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

LEGEND_LABEL_TO_KIND: dict[str, str] = {
    "package": "package",
    "パッケージ": "package",
    "sub-module": "sub_module",
    "sub_module": "sub_module",
    "sub module": "sub_module",
    "module": "sub_module",
    "サブモジュール": "sub_module",
    "interface": "interface",
    "インターフェース": "interface",
    "class": "class",
    "クラス": "class",
    "data class": "data_class",
    "dataclass": "data_class",
    "データクラス": "data_class",
    "free function": "free_function",
    "function": "free_function",
    "フリー関数": "free_function",
    "ros2 node": "ros2_node",
    "ros2ノード": "ros2_node",
    "ROS2ノード": "ros2_node",
    "persistent data": "persistent_data",
    "永続化データ": "persistent_data",
    "memo": "memo",
    "メモ": "memo",
    "ノート": "memo",
}

STYLE_KEYS: tuple[str, ...] = (
    "shape",
    "rounded",
    "container",
    "swimlane",
    "group",
    "backgroundOutline",
    "double",
    "startArrow",
    "endArrow",
    "startFill",
    "endFill",
    "dashed",
)

_CONTENT_RE = re.compile(r'content="([^"]+)"')


@dataclass(frozen=True)
class Node:
    raw_id: str
    label: str
    style: str
    parent: Optional[str]
    is_group: bool
    is_container: bool


@dataclass(frozen=True)
class Edge:
    raw_id: str
    src: str
    dst: str
    label: str
    style: str


@dataclass(frozen=True)
class Graph:
    nodes: list[Node]
    edges: list[Edge]


def _strip_html(s: str) -> str:
    s2: str = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    s2 = re.sub(r"<[^>]+>", "", s2)
    return html.unescape(s2).strip()


def _norm_label(s: str) -> str:
    # normalize NBSP to normal space to avoid "json log" vs "json log" issues
    s2: str = s.replace("\u00a0", " ").replace("\r", "\n")
    s2 = re.sub(r"\s*\n\s*", " ", s2)
    s2 = re.sub(r"\s+", " ", s2)
    return s2.strip()


def _parse_style(style: str) -> dict[str, str]:
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


def _decode_diagram_text(diagram_text: str) -> str:
    s: str = diagram_text.strip()
    if not s:
        raise ValueError("Empty <diagram> text.")
    if "<mxGraphModel" in s:
        return s

    raw: bytes = base64.b64decode(s)
    inflated: Optional[bytes] = None
    for wbits in (-15, 15, 31):
        try:
            inflated = zlib.decompress(raw, wbits=wbits)
            break
        except Exception:
            continue
    if inflated is None:
        raise ValueError("Failed to decompress diagram payload (raw/zlib/gzip).")

    decoded: str = urllib.parse.unquote(inflated.decode("utf-8", errors="strict"))
    if "<mxGraphModel" not in decoded:
        raise ValueError("Decoded payload did not contain <mxGraphModel>.")
    return decoded


def extract_mxfile_xml_from_drawio_svg(svg_text: str) -> str:
    m = _CONTENT_RE.search(svg_text)
    if m is None:
        raise ValueError('No content="...mxfile..." attribute found in SVG.')
    return html.unescape(m.group(1))


def extract_mxgraphmodel_from_file(path: Path, diagram_name: Optional[str] = None) -> str:
    text = path.read_text(encoding="utf-8")
    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".drawio.svg") or path.suffix.lower() == ".svg":
        mxfile_xml = extract_mxfile_xml_from_drawio_svg(text)
        root = ET.fromstring(mxfile_xml)
    else:
        root = ET.fromstring(text)

    if root.tag != "mxfile":
        raise ValueError(f"Expected <mxfile>, got <{root.tag}>.")

    diagrams = list(root.findall("diagram"))
    if not diagrams:
        raise ValueError("No <diagram> found.")
    if diagram_name is None:
        chosen = diagrams[0]
    else:
        chosen = next((d for d in diagrams if d.get("name") == diagram_name), None)
        if chosen is None:
            names = [d.get("name") for d in diagrams]
            raise ValueError(f'No diagram name="{diagram_name}". Available: {names}')

    if chosen.text is None:
        raise ValueError("Chosen <diagram> has no payload.")
    return _decode_diagram_text(chosen.text)


def parse_mxgraphmodel(mxgraph_xml: str) -> Graph:
    root = ET.fromstring(mxgraph_xml)
    nodes: list[Node] = []
    edges: list[Edge] = []

    for cell in root.iter():
        if not cell.tag.endswith("mxCell"):
            continue

        cid = cell.get("id")
        if cid is None:
            continue

        style = cell.get("style", "")
        parent = cell.get("parent")
        value = _norm_label(_strip_html(cell.get("value", "")))

        if cell.get("vertex") == "1":
            s = style.lower()
            is_group = cell.get("group") == "1" or "group" in s
            is_container = "swimlane" in s or "container" in s
            if not value and not (is_group or is_container):
                continue
            nodes.append(
                Node(raw_id=cid, label=value, style=style, parent=parent, is_group=is_group, is_container=is_container)
            )

        elif cell.get("edge") == "1":
            src = cell.get("source")
            dst = cell.get("target")
            if not src or not dst:
                continue
            edges.append(Edge(raw_id=cid, src=src, dst=dst, label=value, style=style))

    return Graph(nodes=nodes, edges=edges)


def normalize_key(label: str) -> str:
    return _norm_label(label).strip().lower()


def build_indexes(nodes: Sequence[Node]) -> tuple[dict[str, Node], dict[str, list[str]]]:
    by_raw: dict[str, Node] = {n.raw_id: n for n in nodes}
    children: dict[str, list[str]] = {}
    for n in nodes:
        if n.parent is None or n.parent in {"0", "1"}:
            continue
        children.setdefault(n.parent, []).append(n.raw_id)
    for pid in children:
        children[pid] = sorted(children[pid], key=lambda x: int(x) if x.isdigit() else x)
    return by_raw, children


def compute_degree(edges: Sequence[Edge]) -> dict[str, int]:
    deg: dict[str, int] = {}
    for e in edges:
        deg[e.src] = deg.get(e.src, 0) + 1
        deg[e.dst] = deg.get(e.dst, 0) + 1
    return deg


def collect_subtree(root_id: str, children: dict[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    stack: list[str] = [root_id]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for ch in children.get(cur, []):
            stack.append(ch)
    return seen


def style_signature(style: str) -> dict[str, str]:
    st = _parse_style(style)
    sig: dict[str, str] = {}
    for k in STYLE_KEYS:
        if k in st:
            sig[k] = st[k]
    return sig


def signature_key(sig: dict[str, str]) -> str:
    return ",".join(f"{k}={sig[k]}" for k in sorted(sig.keys()))


def learn_style_map(
    by_raw: dict[str, Node], children: dict[str, list[str]], legend_root: str
) -> tuple[dict[str, str], list[str]]:
    warnings: list[str] = []
    style_map: dict[str, str] = {}
    for rid in children.get(legend_root, []):
        n = by_raw.get(rid)
        if n is None:
            continue
        kind = LEGEND_LABEL_TO_KIND.get(n.label) or LEGEND_LABEL_TO_KIND.get(normalize_key(n.label))
        if kind is None:
            continue
        sig = style_signature(n.style)
        if not sig:
            warnings.append(f"legend-style-empty: {n.label} (raw_id={n.raw_id})")
            continue
        k = signature_key(sig)
        if k in style_map and style_map[k] != kind:
            warnings.append(f"legend-style-conflict: {k} {style_map[k]} vs {kind}")
        style_map[k] = kind
    if not style_map:
        warnings.append("legend-empty: no legend styles learned")
    return style_map, warnings


def matches(sig_required: dict[str, str], style: str) -> bool:
    st = _parse_style(style)
    for k, v in sig_required.items():
        if st.get(k) != v:
            return False
    return True


def kind_from_style(style: str, learned: dict[str, str]) -> str:
    for key, kind in learned.items():
        req = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in key.split(",") if kv}
        if req and matches(req, style):
            return kind

    st = _parse_style(style)
    s = style.lower()
    shape = (st.get("shape") or "").lower()

    if "container=1" in s and shape in {"folder", "umlframe"}:
        return "sub_module" if shape == "folder" else "package"
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


def infer_rel_from_style(edge_style: str, edge_label: str) -> tuple[str, bool]:
    if edge_label.strip():
        return edge_label.strip(), False

    st = _parse_style(edge_style)
    dashed = st.get("dashed", "0") == "1"
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
    if not inferred or rel != "depends":
        return rel, inferred

    if dst_kind == "persistent_data":
        if src_kind in {"free_function", "ros2_node", "class"}:
            return "writes", True
        return "depends", True

    if src_kind == "persistent_data":
        if dst_kind in {"free_function", "ros2_node", "class"}:
            return "reads", True
        return "depends", True

    if dst_kind == "free_function" and src_kind in {"free_function", "ros2_node", "class"}:
        return "calls", True

    return "depends", True


def sanitize_segment(seg: str) -> str:
    s = seg.strip()
    if not s:
        return "_"
    s = s.replace("/", "／")
    s = re.sub(r"\s+", " ", s)
    return s


def build_path_id(raw_id: str, by_raw: dict[str, Node]) -> str:
    segs: list[str] = []
    cur = by_raw.get(raw_id)
    seen: set[str] = set()
    while cur is not None:
        if cur.raw_id in seen:
            break
        seen.add(cur.raw_id)
        segs.append(sanitize_segment(cur.label) if cur.label else cur.raw_id)
        if cur.parent is None or cur.parent in {"0", "1"}:
            break
        cur = by_raw.get(cur.parent)
    return "/".join(reversed(segs))


def stable_semantic_ids(nodes: Sequence[Node], by_raw: dict[str, Node]) -> dict[str, str]:
    bucket: dict[str, list[str]] = {}
    for n in nodes:
        base = build_path_id(n.raw_id, by_raw)
        bucket.setdefault(base, []).append(n.raw_id)

    out: dict[str, str] = {}
    for base, rids in bucket.items():
        rids_sorted = sorted(rids, key=lambda x: int(x) if x.isdigit() else x)
        if len(rids_sorted) == 1:
            out[rids_sorted[0]] = base
        else:
            for i, rid in enumerate(rids_sorted, start=1):
                out[rid] = f"{base}#{i}"
    return out


def compile_exclude_regexes(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns]


def guess_legend_root(
    by_raw: dict[str, Node],
    children: dict[str, list[str]],
) -> Optional[str]:
    # 1) direct label match
    for rid, n in by_raw.items():
        if normalize_key(n.label) in {"legend", "凡例", "レジェンド"}:
            return rid

    # 2) container with many legend items
    best: Tuple[int, str] | None = None
    for rid, n in by_raw.items():
        if not n.is_container:
            continue
        kids = children.get(rid, [])
        if not kids:
            continue
        hit = 0
        for k in kids:
            c = by_raw.get(k)
            if c is None:
                continue
            if (LEGEND_LABEL_TO_KIND.get(c.label) is not None) or (
                LEGEND_LABEL_TO_KIND.get(normalize_key(c.label)) is not None
            ):
                hit += 1
        if hit >= 4:
            cand = (hit, rid)
            if best is None or cand[0] > best[0]:
                best = cand
    return best[1] if best is not None else None


def is_kind_label(label: str) -> bool:
    return (label in LEGEND_LABEL_TO_KIND) or (normalize_key(label) in LEGEND_LABEL_TO_KIND)


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="drawio_struct_export_json_min.py")
    ap.add_argument("input", help="Input: *.drawio.svg or mxfile XML")
    ap.add_argument("-o", "--output", required=True, help="Output JSON path")
    ap.add_argument("--diagram", default=None, help="Diagram name (if multiple)")
    ap.add_argument("--include-legend", action="store_true", help="Include legend nodes in output")
    ap.add_argument("--exclude-kind", action="append", default=[], help="Exclude nodes of a kind (repeatable)")
    ap.add_argument(
        "--exclude-name-regex", action="append", default=[], help="Exclude nodes whose name matches regex (repeatable)"
    )
    ap.add_argument("--no-rich-relations", action="store_true", help="Disable calls/reads/writes heuristics")
    ap.add_argument(
        "--keep-isolated", action="store_true", help="Keep degree-0 nodes (default: drop non-container isolated nodes)"
    )
    ap.add_argument("--debug", action="store_true", help="Include debug section")
    ap.add_argument("--dump-style-map", default=None, help="Dump learned style map to JSON")
    ap.add_argument("--use-style-map", default=None, help="Load style map from JSON")
    args = ap.parse_args(list(argv))

    mx = extract_mxgraphmodel_from_file(Path(args.input), diagram_name=args.diagram)
    g = parse_mxgraphmodel(mx)
    by_raw, children = build_indexes(g.nodes)
    deg = compute_degree(g.edges)

    # legend detection
    legend_root = guess_legend_root(by_raw, children)
    legend_ids: set[str] = set()
    if legend_root is not None:
        legend_ids = collect_subtree(legend_root, children)

    exclude_kinds = set(args.exclude_kind)
    exclude_name_re = compile_exclude_regexes(list(args.exclude_name_regex))

    warnings: list[str] = []
    learned: dict[str, str]
    if args.use_style_map:
        learned = json.loads(Path(args.use_style_map).read_text(encoding="utf-8"))
    elif legend_root is not None:
        learned, w = learn_style_map(by_raw, children, legend_root)
        warnings.extend(w)
    else:
        learned = {}
        warnings.append("legend-not-found: using heuristics only")

    sem_id = stable_semantic_ids(g.nodes, by_raw)

    raw_kind: dict[str, str] = {n.raw_id: kind_from_style(n.style, learned) for n in g.nodes}

    def name_excluded(name: str) -> bool:
        return any(rx.search(name) for rx in exclude_name_re)

    # filter stats for debug
    dropped = {"legend_subtree": 0, "kind_label_isolated": 0, "isolated": 0, "kind_excluded": 0, "name_excluded": 0}

    def node_excluded(n: Node) -> bool:
        if (not args.include_legend) and (n.raw_id in legend_ids):
            dropped["legend_subtree"] += 1
            return True

        k = raw_kind.get(n.raw_id, "class")
        if k in exclude_kinds:
            dropped["kind_excluded"] += 1
            return True

        if n.label and name_excluded(n.label):
            dropped["name_excluded"] += 1
            return True

        # auto-drop stray kind legend items (degree 0), even if legend root wasn't found
        if (deg.get(n.raw_id, 0) == 0) and (not n.is_container) and is_kind_label(n.label):
            dropped["kind_label_isolated"] += 1
            return True

        # drop isolated non-container nodes by default
        if (not args.keep_isolated) and (deg.get(n.raw_id, 0) == 0) and (not n.is_container):
            dropped["isolated"] += 1
            return True

        return False

    nodes_out: dict[str, str] = {}
    raw_to_sem: dict[str, str] = {}
    for n in g.nodes:
        if node_excluded(n):
            continue
        sid = sem_id[n.raw_id]
        nodes_out[sid] = raw_kind.get(n.raw_id, "class")
        raw_to_sem[n.raw_id] = sid

    edges_out: list[list[Any]] = []
    for e in g.edges:
        if e.src not in raw_to_sem or e.dst not in raw_to_sem:
            continue
        rel, inferred = infer_rel_from_style(e.style, e.label)
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

    if args.debug:
        export["debug"] = {
            "learned_style_map": learned,
            "dropped": dropped,
            "legend_root": sem_id.get(legend_root, None) if legend_root is not None else None,
        }

    Path(args.output).write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dump_style_map:
        Path(args.dump_style_map).write_text(json.dumps(learned, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
