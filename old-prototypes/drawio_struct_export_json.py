#!/usr/bin/env python3
"""
drawio_struct_export_json.py

Convert a draw.io diagram into an LLM-friendly JSON graph.

Input:
  - *.drawio.svg (SVG with embedded mxfile in attribute content="...")
  - mxfile XML (*.drawio / *.xml)

Output JSON fields:
  - meta: {nodes, edges, include_legend}
  - learned_style_signatures: { "<signature>": "<kind>" }
  - nodes: [{id, name, kind, parent, is_container, path}]
  - edges: [{id, src, dst, rel, label, inferred}]
  - warnings: [string...]

Kinds (9):
  package, sub_module, interface, class, data_class, free_function,
  ros2_node, persistent_data, memo
(+ unknown for safety)

Legend-driven typing:
  - If a container with label "legend" exists (case-insensitive), its child items
    are used to learn style->kind signatures.
  - Legend subtree is excluded from output by default.

Usage:
  python drawio_struct_export_json.py input.drawio.svg -o graph.json
"""

from __future__ import annotations

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
from typing import Any, Optional, Sequence


KIND_VALUES: tuple[str, ...] = (
    "package",
    "sub_module",
    "interface",
    "class",
    "data_class",
    "free_function",
    "ros2_node",
    "persistent_data",
    "memo",
    "unknown",
)

LEGEND_LABEL_TO_KIND: dict[str, str] = {
    "package": "package",
    "パッケージ": "package",
    "sub-module": "sub_module",
    "sub_module": "sub_module",
    "sub module": "sub_module",
    "サブモジュール": "sub_module",
    "インターフェース": "interface",
    "interface": "interface",
    "クラス": "class",
    "class": "class",
    "データクラス": "data_class",
    "data class": "data_class",
    "dataclass": "data_class",
    "フリー関数": "free_function",
    "free function": "free_function",
    "function": "free_function",
    "ros2ノード": "ros2_node",
    "ros2 node": "ros2_node",
    "永続化データ": "persistent_data",
    "persistent data": "persistent_data",
    "memo": "memo",
    "メモ": "memo",
    "ノート": "memo",
}

_CONTENT_RE = re.compile(r'content="([^"]+)"')


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    style: str
    parent: Optional[str]
    is_group: bool
    is_container: bool


@dataclass(frozen=True)
class Edge:
    id: str
    src: str
    dst: str
    label: str
    style: str
    parent: Optional[str]


@dataclass(frozen=True)
class DiagramGraph:
    nodes: list[Node]
    edges: list[Edge]
    mxgraphmodel_xml: str


def _strip_html(s: str) -> str:
    s2: str = re.sub(r"<br\\s*/?>", "\\n", s, flags=re.IGNORECASE)
    s2 = re.sub(r"<[^>]+>", "", s2)
    return html.unescape(s2).strip()


def _norm_label(s: str) -> str:
    s2: str = s.replace("\\r", "\\n")
    s2 = re.sub(r"\\s*\\n\\s*", " ", s2)
    s2 = re.sub(r"\\s+", " ", s2)
    return s2.strip()


def _decode_diagram_text(diagram_text: str) -> str:
    s: str = diagram_text.strip()
    if not s:
        raise ValueError("Empty <diagram> text.")
    if "<mxGraphModel" in s:
        return s

    raw: bytes = base64.b64decode(s)
    inflated: Optional[bytes] = None
    for wbits in (-15, 15, 31):  # raw, zlib, gzip
        try:
            inflated = zlib.decompress(raw, wbits=wbits)
            break
        except Exception:
            continue
    if inflated is None:
        raise ValueError("Failed to decompress diagram payload (raw/zlib/gzip).")

    url_encoded: str = inflated.decode("utf-8", errors="strict")
    decoded: str = urllib.parse.unquote(url_encoded)
    if "<mxGraphModel" not in decoded:
        raise ValueError("Decoded payload did not contain <mxGraphModel>.")
    return decoded


def extract_mxfile_xml_from_drawio_svg(svg_text: str) -> str:
    m = _CONTENT_RE.search(svg_text)
    if m is None:
        raise ValueError('No content="...mxfile..." attribute found in SVG.')
    return html.unescape(m.group(1))


def extract_mxgraphmodel_xml_from_mxfile_xml(mxfile_xml: str, diagram_name: Optional[str] = None) -> str:
    root = ET.fromstring(mxfile_xml)
    diagrams = list(root.findall("diagram"))
    if not diagrams:
        raise ValueError("No <diagram> elements found in mxfile.")
    chosen = diagrams[0] if diagram_name is None else next((d for d in diagrams if d.get("name") == diagram_name), None)
    if chosen is None or chosen.text is None:
        raise ValueError("Chosen <diagram> has no text payload.")
    return _decode_diagram_text(chosen.text)


def extract_mxgraphmodel_from_file(path: Path, diagram_name: Optional[str] = None) -> str:
    text: str = path.read_text(encoding="utf-8")
    suffixes: str = "".join(path.suffixes).lower()

    if suffixes.endswith(".drawio.svg") or path.suffix.lower() == ".svg":
        mxfile_xml = extract_mxfile_xml_from_drawio_svg(text)
        return extract_mxgraphmodel_xml_from_mxfile_xml(mxfile_xml, diagram_name=diagram_name)

    if "<mxfile" in text:
        return extract_mxgraphmodel_xml_from_mxfile_xml(text, diagram_name=diagram_name)

    raise ValueError("Unsupported input. Expected .drawio.svg or mxfile XML (.drawio/.xml).")


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


def _is_group_or_container(style: str, cell: ET.Element) -> tuple[bool, bool]:
    s = style.lower()
    is_group = cell.get("group") == "1" or "group" in s
    is_container = "swimlane" in s or "container" in s
    return is_group, is_container


def parse_mxgraphmodel_to_graph(mxgraph_xml: str, *, include_empty_labels: bool = False) -> DiagramGraph:
    root: ET.Element = ET.fromstring(mxgraph_xml)
    nodes: list[Node] = []
    edges: list[Edge] = []

    for cell in root.iter():
        if not cell.tag.endswith("mxCell"):
            continue
        cid = cell.get("id")
        if cid is None:
            continue

        value = _norm_label(_strip_html(cell.get("value", "")))
        style = cell.get("style", "")
        parent = cell.get("parent")

        if cell.get("vertex") == "1":
            if (not include_empty_labels) and (value == ""):
                is_group, is_container = _is_group_or_container(style, cell)
                if not (is_group or is_container):
                    continue
            is_group, is_container = _is_group_or_container(style, cell)
            nodes.append(Node(id=cid, label=value, style=style, parent=parent, is_group=is_group, is_container=is_container))
        elif cell.get("edge") == "1":
            src = cell.get("source")
            dst = cell.get("target")
            if not src or not dst:
                continue
            edges.append(Edge(id=cid, src=src, dst=dst, label=value, style=style, parent=parent))

    return DiagramGraph(nodes=nodes, edges=edges, mxgraphmodel_xml=mxgraph_xml)


def build_node_index(nodes: Sequence[Node]) -> tuple[dict[str, Node], dict[str, list[str]]]:
    by_id: dict[str, Node] = {n.id: n for n in nodes}
    children: dict[str, list[str]] = {}
    for n in nodes:
        if n.parent is None:
            continue
        children.setdefault(n.parent, []).append(n.id)
    for pid, ch in children.items():
        children[pid] = sorted(ch)
    return by_id, children


def ancestors(nid: str, by_id: dict[str, Node]) -> list[str]:
    out: list[str] = []
    cur = by_id.get(nid)
    seen: set[str] = set()
    while cur is not None and cur.parent is not None and cur.parent not in {"0", "1"}:
        if cur.parent in seen:
            break
        seen.add(cur.parent)
        out.append(cur.parent)
        cur = by_id.get(cur.parent)
    return out


def path_labels(nid: str, by_id: dict[str, Node]) -> list[str]:
    a = ancestors(nid, by_id)
    labels: list[str] = []
    for pid in reversed(a):
        p = by_id.get(pid)
        if p is not None and p.label:
            labels.append(p.label)
    n = by_id.get(nid)
    if n is not None and n.label:
        labels.append(n.label)
    return labels


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


def style_signature(style: str) -> str:
    st = _parse_style(style)
    parts: list[str] = []
    for k in STYLE_KEYS:
        if k in st:
            parts.append(f"{k}={st[k]}")
    return ",".join(parts)


def normalize_label_key(label: str) -> str:
    return _norm_label(label).strip().lower()


def find_legend_container_id(by_id: dict[str, Node]) -> Optional[str]:
    for nid, n in by_id.items():
        if normalize_label_key(n.label) == "legend":
            return nid
    return None


def infer_kind_heuristic(n: Node) -> str:
    st = _parse_style(n.style)
    s = n.style.lower()
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
    if st.get("rounded") == "1" or "rounded=1" in s:
        return "free_function"

    return "class"


def learn_style_map_from_legend(g: DiagramGraph, *, legend_container_id: str) -> tuple[dict[str, str], list[str]]:
    by_id, children = build_node_index(g.nodes)
    warnings: list[str] = []
    style_map: dict[str, str] = {}

    for cid in children.get(legend_container_id, []):
        node = by_id.get(cid)
        if node is None:
            continue
        k = LEGEND_LABEL_TO_KIND.get(node.label) or LEGEND_LABEL_TO_KIND.get(normalize_label_key(node.label))
        if k is None:
            continue
        sig = style_signature(node.style)
        if not sig:
            warnings.append(f"legend-style-empty: {node.label} (id={node.id})")
            continue
        if sig in style_map and style_map[sig] != k:
            warnings.append(f"legend-style-conflict: {style_map[sig]} vs {k} for sig={sig}")
        style_map[sig] = k

    if not style_map:
        warnings.append("legend-empty-or-unrecognized: no styles learned")
    return style_map, warnings


def classify_node_kind(n: Node, style_map: dict[str, str]) -> tuple[str, Optional[str]]:
    sig = style_signature(n.style)
    if sig and sig in style_map:
        return style_map[sig], sig
    return infer_kind_heuristic(n), sig if sig else None


def infer_relation(style: str, label: str) -> tuple[str, bool]:
    if label.strip():
        return label.strip(), False

    st = _parse_style(style)
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


def build_export(
    g: DiagramGraph,
    *,
    include_legend: bool,
    style_map: dict[str, str],
    legend_container_id: Optional[str],
) -> dict[str, Any]:
    by_id, children = build_node_index(g.nodes)
    warnings: list[str] = []

    legend_ids: set[str] = set()
    if legend_container_id is not None:
        stack: list[str] = [legend_container_id]
        while stack:
            nid = stack.pop()
            if nid in legend_ids:
                continue
            legend_ids.add(nid)
            for ch in children.get(nid, []):
                stack.append(ch)

    def excluded(nid: str) -> bool:
        return (not include_legend) and (nid in legend_ids)

    out_nodes: list[dict[str, Any]] = []
    for n in g.nodes:
        if excluded(n.id):
            continue
        kind, _sig = classify_node_kind(n, style_map)
        out_nodes.append(
            {
                "id": n.id,
                "name": n.label if n.label else n.id,
                "kind": kind if kind in KIND_VALUES else "unknown",
                "parent": n.parent if (n.parent and n.parent not in {"0", "1"} and not excluded(n.parent)) else None,
                "is_container": bool(n.is_container or n.is_group),
                "path": path_labels(n.id, by_id),
            }
        )

    out_edges: list[dict[str, Any]] = []
    for e in g.edges:
        if excluded(e.src) or excluded(e.dst):
            continue
        rel, inferred = infer_relation(e.style, e.label)
        out_edges.append(
            {
                "id": e.id,
                "src": e.src,
                "dst": e.dst,
                "rel": rel,
                "label": e.label if e.label else None,
                "inferred": inferred,
            }
        )

    return {
        "meta": {"nodes": len(out_nodes), "edges": len(out_edges), "include_legend": include_legend},
        "learned_style_signatures": style_map,
        "nodes": out_nodes,
        "edges": out_edges,
        "warnings": warnings,
    }


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="drawio_struct_export_json.py")
    ap.add_argument("input", help="Input: *.drawio.svg or mxfile *.drawio/*.xml")
    ap.add_argument("-o", "--output", required=True, help="Output JSON path")
    ap.add_argument("--diagram", default=None, help='Diagram name (if multiple <diagram name="...">)')
    ap.add_argument("--include-legend", action="store_true", help="Include legend nodes in output")
    ap.add_argument("--include-empty-labels", action="store_true", help="Include unlabeled vertex cells too (debug)")
    ap.add_argument("--dump-style-map", default=None, help="Dump learned style signature map (json)")
    ap.add_argument("--use-style-map", default=None, help="Load style signature map (json) instead of learning")
    args = ap.parse_args(list(argv))

    mx = extract_mxgraphmodel_from_file(Path(args.input), diagram_name=args.diagram)
    g = parse_mxgraphmodel_to_graph(mx, include_empty_labels=args.include_empty_labels)

    by_id, _children = build_node_index(g.nodes)
    legend_id = find_legend_container_id(by_id)

    warnings: list[str] = []
    if args.use_style_map:
        style_map = json.loads(Path(args.use_style_map).read_text(encoding="utf-8"))
    elif legend_id is not None:
        style_map, w = learn_style_map_from_legend(g, legend_container_id=legend_id)
        warnings.extend(w)
    else:
        style_map = {}
        warnings.append("legend-not-found: falling back to heuristics")

    export = build_export(g, include_legend=args.include_legend, style_map=style_map, legend_container_id=legend_id)
    export["warnings"] = warnings + export.get("warnings", [])

    Path(args.output).write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dump_style_map:
        Path(args.dump_style_map).write_text(json.dumps(style_map, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
