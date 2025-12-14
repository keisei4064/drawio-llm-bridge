#!/usr/bin/env python3
"""
drawio_struct_export_fixed.py

This version fixes the Mermaid parse error you hit ("flowchart LR\n ..."):
- Mermaid output uses REAL newlines (not the literal sequence "\n")
- Edge relation labels are always quoted: -->|"rel"| (safe for spaces/symbols)
- Mermaid IDs never start with digits (prefix n_)
- Containers are subgraphs only (no duplicate container-as-node)
- Labels: newlines collapsed to spaces
- Relation inference from edge style: inherits/implements/composes/aggregates, else depends

Default output: Markdown containing exactly one Mermaid code block.
Input: .drawio.svg (mxfile embedded) or mxfile XML (.drawio/.xml)

Python: 3.10+
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
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


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    style: str
    parent: Optional[str]
    is_group: bool
    is_container: bool
    is_folder_like: bool


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


_CONTENT_RE = re.compile(r'content="([^"]+)"')


def _strip_html(s: str) -> str:
    s2: str = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    s2 = re.sub(r"<[^>]+>", "", s2)
    return html.unescape(s2).strip()


def _norm_label(s: str) -> str:
    s2: str = s.replace("\r", "\n")
    s2 = re.sub(r"\s*\n\s*", " ", s2)
    s2 = re.sub(r"\s+", " ", s2)
    return s2.strip()


def _decode_diagram_text(diagram_text: str) -> str:
    s: str = diagram_text.strip()
    if not s:
        raise ValueError("Empty <diagram> text.")
    if "<mxGraphModel" in s:
        return s
    if "%3CmxGraphModel" in s or "%3Cmxfile" in s:
        u = urllib.parse.unquote(s)
        if "<mxGraphModel" in u:
            return u

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

    if "<mxfile" in decoded and "<mxGraphModel" not in decoded:
        mxfile_root = ET.fromstring(decoded)
        diagram_el = mxfile_root.find("diagram")
        if diagram_el is None or diagram_el.text is None:
            raise ValueError("Decoded mxfile had no <diagram> text.")
        return _decode_diagram_text(diagram_el.text)

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
    if root.tag != "mxfile":
        raise ValueError(f"Expected <mxfile> root, got <{root.tag}>.")
    diagrams = list(root.findall("diagram"))
    if not diagrams:
        raise ValueError("No <diagram> elements found in mxfile.")

    chosen: ET.Element
    if diagram_name is None:
        chosen = diagrams[0]
    else:
        match = [d for d in diagrams if d.get("name") == diagram_name]
        if not match:
            names = [d.get("name") for d in diagrams]
            raise ValueError(f'No diagram with name="{diagram_name}". Available: {names}')
        chosen = match[0]
    if chosen.text is None:
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


def _is_folder_like(style: str, label: str) -> bool:
    s = style.lower()
    if "shape=folder" in s:
        return True
    if "folder" in s and "shape=" in s:
        return True
    if label and len(label) <= 32 and ("swimlane" in s or "group" in s):
        return True
    return False


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
            nodes.append(
                Node(
                    id=cid,
                    label=value,
                    style=style,
                    parent=parent,
                    is_group=is_group,
                    is_container=is_container,
                    is_folder_like=_is_folder_like(style, value),
                )
            )
        elif cell.get("edge") == "1":
            src = cell.get("source")
            dst = cell.get("target")
            if not src or not dst:
                continue
            edges.append(Edge(id=cid, src=src, dst=dst, label=value, style=style, parent=parent))

    return DiagramGraph(nodes=nodes, edges=edges, mxgraphmodel_xml=mxgraph_xml)


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


def _infer_relation_from_style(style: str) -> str:
    st = _parse_style(style)
    dashed = st.get("dashed", "0") == "1"
    start_arrow = (st.get("startArrow") or "").lower()
    end_arrow = (st.get("endArrow") or "").lower()
    start_fill = st.get("startFill", "1")
    end_fill = st.get("endFill", "1")

    if "diamond" in start_arrow:
        return "composes" if start_fill == "1" else "aggregates"

    triangle_like = end_arrow in {"block", "blockthin", "triangle", "open", "classic"}
    if triangle_like and end_fill == "0":
        return "implements" if dashed else "inherits"

    return "depends"


def _is_container_node(n: Node) -> bool:
    return n.is_folder_like or n.is_container or n.is_group


def _escape_mermaid_label(label: str) -> str:
    return label.replace('"', '\\"')


def _mermaid_id_token(s: str) -> str:
    t = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    t = re.sub(r"_+", "_", t).strip("_")
    if not t:
        t = "node"
    if t[0].isdigit():
        t = "n_" + t
    return t


def _build_parent_index(nodes: Sequence[Node]) -> tuple[dict[str, Node], dict[str, list[Node]]]:
    by_id: dict[str, Node] = {n.id: n for n in nodes}
    children: dict[str, list[Node]] = {}
    for n in nodes:
        if n.parent is None:
            continue
        children.setdefault(n.parent, []).append(n)
    for pid, ch in children.items():
        children[pid] = sorted(ch, key=lambda x: ((x.label or "").lower(), x.id))
    return by_id, children


def _container_path_labels(nid: str, by_id: dict[str, Node]) -> list[str]:
    out: list[str] = []
    cur = by_id.get(nid)
    seen: set[str] = set()
    while cur is not None and cur.parent is not None and cur.parent not in {"0", "1"}:
        if cur.parent in seen:
            break
        seen.add(cur.parent)
        p = by_id.get(cur.parent)
        if p is None:
            break
        if _is_container_node(p) and p.label:
            out.append(p.label)
        cur = p
    out.reverse()
    return out


def _make_human_key(n: Node, by_id: dict[str, Node]) -> str:
    parts = _container_path_labels(n.id, by_id)
    base = n.label if n.label else n.id
    parts.append(base)
    return "__".join(parts)


def _assign_mermaid_ids(nodes: Sequence[Node]) -> dict[str, str]:
    by_id, _ = _build_parent_index(nodes)
    used: dict[str, int] = {}
    mapping: dict[str, str] = {}
    for n in sorted(nodes, key=lambda x: (_make_human_key(x, by_id).lower(), x.id)):
        key = _make_human_key(n, by_id)
        tok = _mermaid_id_token(key)
        k = tok.lower()
        used[k] = used.get(k, 0) + 1
        if used[k] > 1:
            tok = f"{tok}_{used[k]}"
        mapping[n.id] = tok
    return mapping


def _ensure_edge_endpoints_present(g: DiagramGraph) -> tuple[DiagramGraph, list[str]]:
    by_id: dict[str, Node] = {n.id: n for n in g.nodes}
    warnings: list[str] = []
    extra: list[Node] = []

    def add_placeholder(nid: str) -> None:
        if nid in by_id:
            return
        warnings.append(f"missing-node: referenced id={nid} (added placeholder)")
        ph = Node(
            id=nid,
            label=f"UNKNOWN {nid}",
            style="",
            parent=None,
            is_group=False,
            is_container=False,
            is_folder_like=False,
        )
        by_id[nid] = ph
        extra.append(ph)

    for e in g.edges:
        add_placeholder(e.src)
        add_placeholder(e.dst)

    if not extra:
        return g, warnings
    return DiagramGraph(
        nodes=list(g.nodes) + extra, edges=list(g.edges), mxgraphmodel_xml=g.mxgraphmodel_xml
    ), warnings


def graph_to_mermaid(g: DiagramGraph) -> tuple[str, list[str]]:
    g2, warnings = _ensure_edge_endpoints_present(g)
    by_id, children = _build_parent_index(g2.nodes)
    mid_of = _assign_mermaid_ids(g2.nodes)

    lines: list[str] = ["flowchart LR"]
    declared_nodes: set[str] = set()
    declared_clusters: set[str] = set()

    def emit_node(n: Node, indent: int) -> None:
        if _is_container_node(n):
            return
        mid = mid_of[n.id]
        if mid in declared_nodes:
            return
        label = n.label if n.label else n.id
        label2 = f"{label} [id={n.id}]"
        lines.append("  " * indent + f'{mid}["{_escape_mermaid_label(label2)}"]')
        declared_nodes.add(mid)

    def emit_subtree(n: Node, indent: int, stack: set[str]) -> None:
        if n.id in stack:
            emit_node(n, indent)
            return
        stack2 = set(stack)
        stack2.add(n.id)

        if _is_container_node(n) and n.id in children:
            cluster_id = f"cluster_{mid_of[n.id]}"
            if cluster_id in declared_clusters:
                return
            declared_clusters.add(cluster_id)
            label = n.label if n.label else n.id
            lines.append("  " * indent + f'subgraph {cluster_id}["{_escape_mermaid_label(label)}"]')
            for c in children.get(n.id, []):
                emit_subtree(c, indent + 1, stack2)
            lines.append("  " * indent + "end")
        else:
            emit_node(n, indent)
            if n.id in children:
                cluster_id = f"cluster_{mid_of[n.id]}"
                if cluster_id not in declared_clusters:
                    declared_clusters.add(cluster_id)
                    label = n.label if n.label else n.id
                    lines.append("  " * indent + f'subgraph {cluster_id}["{_escape_mermaid_label(label)} (children)"]')
                    for c in children.get(n.id, []):
                        emit_subtree(c, indent + 1, stack2)
                    lines.append("  " * indent + "end")

    roots: list[Node] = []
    for n in g2.nodes:
        if n.parent is None or n.parent not in by_id or n.parent in {"0", "1"}:
            roots.append(n)
    roots = sorted(roots, key=lambda x: ((x.label or "").lower(), x.id))
    for r in roots:
        emit_subtree(r, 1, set())

    for n in sorted(g2.nodes, key=lambda x: ((x.label or "").lower(), x.id)):
        emit_node(n, 1)

    for e in g2.edges:
        s = mid_of[e.src]
        t = mid_of[e.dst]
        rel = e.label if e.label else _infer_relation_from_style(e.style)
        rel = _norm_label(rel) if rel else "depends"
        rel = rel if rel else "depends"
        lines.append(f'  {s} -->|"{_escape_mermaid_label(rel)}"| {t}')

    return "\n".join(lines) + "\n", warnings


def graph_to_markdown(g: DiagramGraph, *, title: str, include_stats: bool, include_warnings: bool) -> str:
    mermaid, warnings = graph_to_mermaid(g)
    out: list[str] = [f"# {title}", ""]
    if include_stats:
        out += [f"- nodes: {len(g.nodes)}", f"- edges: {len(g.edges)}", ""]
    if include_warnings and warnings:
        out.append("## Warnings")
        out += [f"- {w}" for w in warnings]
        out.append("")
    out.append("```mermaid")
    out.append(mermaid.rstrip())
    out.append("```")
    out.append("")
    return "\n".join(out)


def graph_to_json_obj(g: DiagramGraph) -> dict[str, Any]:
    return {"nodes": [dataclasses.asdict(n) for n in g.nodes], "edges": [dataclasses.asdict(e) for e in g.edges]}


def main(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(prog="drawio_struct_export_fixed.py")
    p.add_argument("input", help="Input: *.drawio.svg or mxfile *.drawio/*.xml")
    p.add_argument("-o", "--output", default=None, help="Output Markdown path (default: stdout)")
    p.add_argument("--title", default=None, help="Markdown title (default: input filename)")
    p.add_argument("--diagram", default=None, help='Diagram name (if multiple <diagram name="...">)')
    p.add_argument("--include-empty-labels", action="store_true", help="Include unlabeled vertex cells too")
    p.add_argument("--no-stats", action="store_true", help="Omit node/edge counts")
    p.add_argument("--no-warnings", action="store_true", help="Omit warnings section")
    p.add_argument("--dump-mxgraphmodel", default=None, help="Write extracted mxGraphModel XML (debug)")
    p.add_argument("--dump-json", default=None, help="Write raw nodes/edges JSON (debug)")
    args = p.parse_args(list(argv))

    in_path = Path(args.input)
    mx = extract_mxgraphmodel_from_file(in_path, diagram_name=args.diagram)
    g = parse_mxgraphmodel_to_graph(mx, include_empty_labels=args.include_empty_labels)

    title = args.title if args.title else in_path.name
    md = graph_to_markdown(g, title=title, include_stats=(not args.no_stats), include_warnings=(not args.no_warnings))

    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
    else:
        sys.stdout.write(md)

    if args.dump_mxgraphmodel:
        Path(args.dump_mxgraphmodel).write_text(g.mxgraphmodel_xml, encoding="utf-8")
    if args.dump_json:
        Path(args.dump_json).write_text(
            json.dumps(graph_to_json_obj(g), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
