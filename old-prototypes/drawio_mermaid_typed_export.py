#!/usr/bin/env python3
"""
drawio_mermaid_typed_export.py

Export a draw.io diagram (.drawio.svg or mxfile XML) into a single Markdown file
containing one Mermaid flowchart code block, preserving hierarchy (parent/child
containers as nested subgraphs) and classifying nodes into a fixed set of kinds.

Primary goal: be LLM-friendly while staying deterministic and reusable.

Supported node kinds (9):
- interface
- class
- ros2_node
- free_function
- data_class
- persistent_data
- package
- sub_module
- memo

How classification works (robust approach):
1) If the diagram contains "legend" nodes whose labels equal any kind name
   (e.g., "インターフェース", "クラス", "ROS2ノード", "package", "memo", "module"/"sub-module"),
   we learn style-signatures from them and classify other nodes by signature match.
2) Otherwise, we fall back to simple heuristics (shape/rounded/cylinder/etc).

Edges:
- Always labeled. Prefer explicit edge labels; otherwise infer from edge style
  (inherits/implements/composes/aggregates/depends).

Output:
- Markdown with a single ```mermaid``` block.
- Node labels include kind tags like:  Name <br/>[class]

Python 3.10+
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
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence


# -----------------------------
# Model
# -----------------------------
class SemanticKind(str, Enum):
    INTERFACE = "interface"
    CLASS = "class"
    ROS2_NODE = "ros2_node"
    FREE_FUNCTION = "free_function"
    DATA_CLASS = "data_class"
    PERSISTENT_DATA = "persistent_data"
    PACKAGE = "package"
    SUB_MODULE = "sub_module"
    MEMO = "memo"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Node:
    id: str
    label: str
    style: str
    parent: Optional[str]


@dataclass(frozen=True, slots=True)
class Edge:
    id: str
    src: str
    dst: str
    label: str
    style: str
    parent: Optional[str]


@dataclass(frozen=True, slots=True)
class DiagramGraph:
    nodes: list[Node]
    edges: list[Edge]
    mxgraphmodel_xml: str


# -----------------------------
# Utils
# -----------------------------
_CONTENT_RE = re.compile(r'content="([^"]+)"')


def _strip_html(s: str) -> str:
    s2 = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    s2 = re.sub(r"<[^>]+>", "", s2)
    return html.unescape(s2).strip()


def _norm_label(s: str) -> str:
    s2 = s.replace("\r", "\n")
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


# -----------------------------
# Decode draw.io payloads
# -----------------------------
def _decode_diagram_text(diagram_text: str) -> str:
    s = diagram_text.strip()
    if not s:
        raise ValueError("Empty <diagram> text.")
    if "<mxGraphModel" in s:
        return s

    # URL-encoded XML without compression?
    if "%3CmxGraphModel" in s or "%3Cmxfile" in s:
        u = urllib.parse.unquote(s)
        if "<mxGraphModel" in u:
            return u

    raw = base64.b64decode(s)
    inflated: Optional[bytes] = None
    for wbits in (-15, 15, 31):  # raw, zlib, gzip
        try:
            inflated = zlib.decompress(raw, wbits=wbits)
            break
        except Exception:
            continue
    if inflated is None:
        raise ValueError("Failed to decompress diagram payload (raw/zlib/gzip).")

    url_encoded = inflated.decode("utf-8", errors="strict")
    decoded = urllib.parse.unquote(url_encoded)

    # Sometimes decoded is an mxfile again
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


def extract_mxgraphmodel_xml_from_mxfile_xml(mxfile_xml: str, diagram_name: Optional[str]) -> str:
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


def extract_mxgraphmodel_from_file(path: Path, diagram_name: Optional[str]) -> str:
    text = path.read_text(encoding="utf-8")
    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".drawio.svg") or path.suffix.lower() == ".svg":
        mxfile_xml = extract_mxfile_xml_from_drawio_svg(text)
        return extract_mxgraphmodel_xml_from_mxfile_xml(mxfile_xml, diagram_name)

    if "<mxfile" in text:
        return extract_mxgraphmodel_xml_from_mxfile_xml(text, diagram_name)

    raise ValueError("Unsupported input. Expected .drawio.svg or mxfile XML (.drawio/.xml).")


# -----------------------------
# Parse mxGraphModel -> graph
# -----------------------------
def parse_mxgraphmodel_to_graph(mxgraph_xml: str) -> DiagramGraph:
    root = ET.fromstring(mxgraph_xml)
    nodes: list[Node] = []
    edges: list[Edge] = []

    for cell in root.iter():
        if not cell.tag.endswith("mxCell"):
            continue

        at = cell.attrib
        cid = at.get("id")
        if cid is None:
            continue

        style = at.get("style", "")
        parent = at.get("parent")
        value = _norm_label(_strip_html(at.get("value", "")))

        if at.get("vertex") == "1":
            # keep even empty labels (containers may be empty)
            nodes.append(Node(id=cid, label=value, style=style, parent=parent))

        elif at.get("edge") == "1":
            src = at.get("source")
            dst = at.get("target")
            if not src or not dst:
                continue
            edges.append(Edge(id=cid, src=src, dst=dst, label=value, style=style, parent=parent))

    return DiagramGraph(nodes=nodes, edges=edges, mxgraphmodel_xml=mxgraph_xml)


# -----------------------------
# Containers (hierarchy)
# -----------------------------
def _is_container_node(style: str) -> bool:
    st = _parse_style(style)
    # draw.io containers often set container=1 or swimlane=1
    if st.get("container") == "1":
        return True
    if st.get("swimlane") == "1":
        return True
    # Also treat folder/umlFrame as containers in our convention
    shp = (st.get("shape") or "").lower()
    return shp in {"folder", "umlframe"}


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


# -----------------------------
# Classification (9 kinds)
# -----------------------------
_KIND_LABELS: dict[str, SemanticKind] = {
    # Japanese legend labels
    "インターフェース": SemanticKind.INTERFACE,
    "クラス": SemanticKind.CLASS,
    "ROS2ノード": SemanticKind.ROS2_NODE,
    "フリー関数": SemanticKind.FREE_FUNCTION,
    "データクラス": SemanticKind.DATA_CLASS,
    "永続化データ": SemanticKind.PERSISTENT_DATA,
    "package": SemanticKind.PACKAGE,
    "memo": SemanticKind.MEMO,
    "sub-module": SemanticKind.SUB_MODULE,
    "sub_module": SemanticKind.SUB_MODULE,
    "module": SemanticKind.SUB_MODULE,  # in your template, legend says "module"
    # English aliases
    "interface": SemanticKind.INTERFACE,
    "class": SemanticKind.CLASS,
    "ros2_node": SemanticKind.ROS2_NODE,
    "free_function": SemanticKind.FREE_FUNCTION,
    "data_class": SemanticKind.DATA_CLASS,
    "persistent_data": SemanticKind.PERSISTENT_DATA,
}


@dataclass(frozen=True, slots=True)
class KindSignature:
    kind: SemanticKind
    # required key-value pairs for a match (small but distinctive)
    required: tuple[tuple[str, str], ...]


def _is_legend_container_label(label: str) -> bool:
    """Return True if this node should be treated as a legend container (excluded from output).

    We keep this intentionally strict to avoid accidentally hiding real nodes.
    """
    if not label:
        return False
    s = _norm_label(label).strip().lower()
    return s in {"legend"}


def _collect_subtree_ids(root_id: str, children: dict[str, list[Node]]) -> set[str]:
    out: set[str] = set()
    stack: list[str] = [root_id]
    while stack:
        cur = stack.pop()
        if cur in out:
            continue
        out.add(cur)
        for ch in children.get(cur, []):
            stack.append(ch.id)
    return out


def _find_legend_node_ids(nodes: Sequence[Node]) -> set[str]:
    """Find node ids belonging to legend container(s) and their descendants.

    These nodes are still used for learning style signatures, but removed from the exported graph.
    """
    _, children = _build_parent_index(nodes)
    legend_roots = [n.id for n in nodes if _is_legend_container_label(n.label)]
    out: set[str] = set()
    for rid in legend_roots:
        out |= _collect_subtree_ids(rid, children)
    return out


def _signature_from_style(kind: SemanticKind, style: str) -> KindSignature:
    st = _parse_style(style)
    shp = st.get("shape", "")
    req: list[tuple[str, str]] = []

    # Prefer "shape" if present: it is the most stable discriminator.
    if shp:
        req.append(("shape", shp))

    # Add a few extra discriminators that help with collisions.
    for k in ("rounded", "backgroundOutline", "container", "swimlane"):
        if k in st:
            req.append((k, st[k]))

    # Fallback: if no shape, rounded is typically the key difference
    if not shp and "rounded" in st:
        req.append(("rounded", st["rounded"]))

    # Deduplicate while preserving order
    seen: set[tuple[str, str]] = set()
    req2: list[tuple[str, str]] = []
    for kv in req:
        if kv not in seen:
            seen.add(kv)
            req2.append(kv)

    return KindSignature(kind=kind, required=tuple(req2))


def _learn_signatures_from_legend(nodes: Sequence[Node]) -> dict[SemanticKind, KindSignature]:
    learned: dict[SemanticKind, KindSignature] = {}
    for n in nodes:
        if n.label in _KIND_LABELS:
            k = _KIND_LABELS[n.label]
            learned[k] = _signature_from_style(k, n.style)
    return learned


def _matches_signature(style: str, sig: KindSignature) -> bool:
    st = _parse_style(style)
    for k, v in sig.required:
        if st.get(k) != v:
            return False
    return True


def _classify_fallback(node: Node) -> SemanticKind:
    st = _parse_style(node.style)
    shp = (st.get("shape") or "").lower()

    if shp == "hexagon":
        return SemanticKind.INTERFACE
    if shp in {"cylinder", "cylinder3"}:
        return SemanticKind.PERSISTENT_DATA
    if shp in {"note", "note2"}:
        return SemanticKind.MEMO
    if shp == "umlframe":
        return SemanticKind.PACKAGE
    if shp == "folder":
        return SemanticKind.SUB_MODULE
    if shp in {"cube", "box3d"}:
        return SemanticKind.DATA_CLASS
    if shp == "process" and st.get("backgroundOutline") == "1":
        return SemanticKind.ROS2_NODE

    # Plain rectangles: rounded=1 => free function, rounded=0 => class
    if st.get("rounded") == "1":
        return SemanticKind.FREE_FUNCTION
    if st.get("rounded") == "0":
        return SemanticKind.CLASS

    # Pure text annotations (title etc.)
    if shp == "text" or st.get("strokeColor") == "none":
        return SemanticKind.MEMO

    return SemanticKind.UNKNOWN


def classify_nodes(
    nodes: Sequence[Node],
    *,
    learn_from_legend: bool,
) -> tuple[dict[str, SemanticKind], dict[SemanticKind, KindSignature]]:
    learned: dict[SemanticKind, KindSignature] = _learn_signatures_from_legend(nodes) if learn_from_legend else {}
    kind_of: dict[str, SemanticKind] = {}

    for n in nodes:
        # If the node itself is a legend label, classify it by its label (always)
        if n.label in _KIND_LABELS:
            kind_of[n.id] = _KIND_LABELS[n.label]
            continue

        # Signature-based classification
        picked: Optional[SemanticKind] = None
        if learned:
            for k, sig in learned.items():
                if _matches_signature(n.style, sig):
                    picked = k
                    break

        kind_of[n.id] = picked if picked is not None else _classify_fallback(n)

    return kind_of, learned


# -----------------------------
# Relation inference (edge style)
# -----------------------------
def _infer_relation_from_style(style: str) -> str:
    """
    Heuristics for draw.io edge styles (UML-ish):
    - composes/aggregates: diamond at either end (prefer startArrow; handle endArrow too)
    - inherits/implements: hollow triangle (endFill=0) at target; dashed => implements
    - else: depends
    """
    st = _parse_style(style)
    dashed = st.get("dashed", "0") == "1"
    start_arrow = (st.get("startArrow") or "").lower()
    end_arrow = (st.get("endArrow") or "").lower()
    start_fill = st.get("startFill", "1")
    end_fill = st.get("endFill", "1")

    # Diamonds at source/target
    if "diamond" in start_arrow:
        return "composes" if start_fill == "1" else "aggregates"
    if "diamond" in end_arrow:
        # draw.io often uses endArrow=diamondThin; treat same label (direction ambiguity is acceptable here)
        return "composes" if end_fill == "1" else "aggregates"

    # Hollow triangle at target => inheritance-ish
    triangle_like = end_arrow in {"block", "blockthin", "triangle", "open", "classic"}
    hollow = end_fill == "0"
    if triangle_like and hollow:
        return "implements" if dashed else "inherits"

    return "depends"


# -----------------------------
# Mermaid generation
# -----------------------------
def _ensure_edge_endpoints_present(g: DiagramGraph) -> tuple[DiagramGraph, list[str]]:
    by_id: dict[str, Node] = {n.id: n for n in g.nodes}
    warnings: list[str] = []
    extra: list[Node] = []

    def add_placeholder(nid: str) -> None:
        if nid in by_id:
            return
        warnings.append(f"missing-node: referenced id={nid} (added placeholder)")
        ph = Node(id=nid, label=f"UNKNOWN {nid}", style="", parent=None)
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


def _container_path_labels(nid: str, by_id: dict[str, Node], kind_of: dict[str, SemanticKind]) -> list[str]:
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
        if _is_container_node(p.style) and p.label:
            out.append(p.label)
        cur = p
    out.reverse()
    return out


def _make_human_key(n: Node, by_id: dict[str, Node], kind_of: dict[str, SemanticKind]) -> str:
    parts = _container_path_labels(n.id, by_id, kind_of)
    base = n.label if n.label else n.id
    parts.append(base)
    return "__".join(parts)


def _assign_mermaid_ids(nodes: Sequence[Node], kind_of: dict[str, SemanticKind]) -> dict[str, str]:
    by_id, _ = _build_parent_index(nodes)
    used: dict[str, int] = {}
    mapping: dict[str, str] = {}
    for n in sorted(nodes, key=lambda x: (_make_human_key(x, by_id, kind_of).lower(), x.id)):
        key = _make_human_key(n, by_id, kind_of)
        tok = _mermaid_id_token(key)
        k = tok.lower()
        used[k] = used.get(k, 0) + 1
        if used[k] > 1:
            tok = f"{tok}_{used[k]}"
        mapping[n.id] = tok
    return mapping


def graph_to_mermaid(
    g: DiagramGraph,
    *,
    kind_of: dict[str, SemanticKind],
    include_ids: bool,
    include_kind_tags: bool,
) -> tuple[str, list[str]]:
    g2, warnings = _ensure_edge_endpoints_present(g)
    by_id, children = _build_parent_index(g2.nodes)
    mid_of = _assign_mermaid_ids(g2.nodes, kind_of)

    lines: list[str] = ["flowchart LR"]
    declared_nodes: set[str] = set()
    declared_clusters: set[str] = set()

    def mk_label(n: Node) -> str:
        base = n.label if n.label else n.id
        k = kind_of.get(n.id, SemanticKind.UNKNOWN)
        parts: list[str] = [base]
        if include_kind_tags and k != SemanticKind.UNKNOWN:
            parts.append(f"[{k.value}]")
        if include_ids:
            parts.append(f"(id={n.id})")
        # Mermaid supports <br/> in labels.
        return "<br/>".join(parts)

    def emit_node(n: Node, indent: int) -> None:
        if _is_container_node(n.style):
            return
        mid = mid_of[n.id]
        if mid in declared_nodes:
            return
        label2 = mk_label(n)
        lines.append("  " * indent + f'{mid}["{_escape_mermaid_label(label2)}"]')
        declared_nodes.add(mid)

    def emit_subtree(n: Node, indent: int, stack: set[str]) -> None:
        if n.id in stack:
            emit_node(n, indent)
            return
        stack2 = set(stack)
        stack2.add(n.id)

        if _is_container_node(n.style) and n.id in children:
            cluster_id = f"cluster_{mid_of[n.id]}"
            if cluster_id in declared_clusters:
                return
            declared_clusters.add(cluster_id)
            label = mk_label(n)
            lines.append("  " * indent + f'subgraph {cluster_id}["{_escape_mermaid_label(label)}"]')
            for c in children.get(n.id, []):
                emit_subtree(c, indent + 1, stack2)
            lines.append("  " * indent + "end")
        else:
            emit_node(n, indent)
            # non-container with children is rare; still show its children under it
            if n.id in children:
                cluster_id = f"cluster_{mid_of[n.id]}"
                if cluster_id not in declared_clusters:
                    declared_clusters.add(cluster_id)
                    label = mk_label(n) + "<br/>(children)"
                    lines.append("  " * indent + f'subgraph {cluster_id}["{_escape_mermaid_label(label)}"]')
                    for c in children.get(n.id, []):
                        emit_subtree(c, indent + 1, stack2)
                    lines.append("  " * indent + "end")

    # roots: parent missing/None or parent in {0,1}
    roots: list[Node] = []
    for n in g2.nodes:
        if n.parent is None or n.parent not in by_id or n.parent in {"0", "1"}:
            roots.append(n)
    roots = sorted(roots, key=lambda x: ((x.label or "").lower(), x.id))
    for r in roots:
        emit_subtree(r, 1, set())

    # Ensure orphan non-container nodes exist
    for n in sorted(g2.nodes, key=lambda x: ((x.label or "").lower(), x.id)):
        emit_node(n, 1)

    # Edges with relation labels (prefer explicit label)
    for e in g2.edges:
        s = mid_of[e.src]
        t = mid_of[e.dst]
        rel = e.label if e.label else _infer_relation_from_style(e.style)
        rel = _norm_label(rel) if rel else "depends"
        rel = rel if rel else "depends"
        lines.append(f'  {s} -->|"{_escape_mermaid_label(rel)}"| {t}')

    return "\n".join(lines) + "\n", warnings


# -----------------------------
# Markdown / JSON output
# -----------------------------
def graph_to_markdown(
    g: DiagramGraph,
    *,
    title: str,
    kind_of: dict[str, SemanticKind],
    include_stats: bool,
    include_warnings: bool,
    include_ids: bool,
    include_kind_tags: bool,
    learned: dict[SemanticKind, KindSignature],
) -> str:
    mermaid, warnings = graph_to_mermaid(
        g,
        kind_of=kind_of,
        include_ids=include_ids,
        include_kind_tags=include_kind_tags,
    )

    out: list[str] = [f"# {title}", ""]
    if include_stats:
        out += [f"- nodes: {len(g.nodes)}", f"- edges: {len(g.edges)}", ""]

    if learned:
        out.append("## Learned style signatures")
        for k in sorted(learned.keys(), key=lambda x: x.value):
            sig = learned[k]
            req = ", ".join([f"{a}={b}" for a, b in sig.required]) if sig.required else "(none)"
            out.append(f"- {k.value}: {req}")
        out.append("")

    if include_warnings and warnings:
        out.append("## Warnings")
        out += [f"- {w}" for w in warnings]
        out.append("")

    out.append("```mermaid")
    out.append(mermaid.rstrip())
    out.append("```")
    out.append("")
    return "\n".join(out)


def graph_to_json_obj(g: DiagramGraph, kind_of: dict[str, SemanticKind]) -> dict[str, Any]:
    nodes_out: list[dict[str, Any]] = []
    for n in g.nodes:
        nodes_out.append(
            {
                **dataclasses.asdict(n),
                "kind": kind_of.get(n.id, SemanticKind.UNKNOWN).value,
            }
        )
    edges_out = [dataclasses.asdict(e) for e in g.edges]
    return {"nodes": nodes_out, "edges": edges_out}


# -----------------------------
# CLI
# -----------------------------
def main(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(prog="drawio_mermaid_typed_export.py")
    p.add_argument("input", help="Input: *.drawio.svg or mxfile *.drawio/*.xml")
    p.add_argument("-o", "--output", default=None, help="Output Markdown path (default: stdout)")
    p.add_argument("--title", default=None, help="Markdown title (default: input filename)")
    p.add_argument("--diagram", default=None, help='Diagram name (if multiple <diagram name="...">)')
    p.add_argument("--no-stats", action="store_true", help="Omit node/edge counts")
    p.add_argument("--no-warnings", action="store_true", help="Omit warnings section")
    p.add_argument("--no-kind-tags", action="store_true", help="Do not add [kind] tags to labels")
    p.add_argument("--no-ids", action="store_true", help="Do not add (id=...) tags to labels")
    p.add_argument("--no-learn-legend", action="store_true", help="Do not learn signatures from legend nodes")
    p.add_argument("--dump-json", default=None, help="Write nodes/edges/kinds JSON (debug)")
    args = p.parse_args(list(argv))

    in_path = Path(args.input)
    mx = extract_mxgraphmodel_from_file(in_path, diagram_name=args.diagram)
    g = parse_mxgraphmodel_to_graph(mx)

    kind_of, learned = classify_nodes(g.nodes, learn_from_legend=(not args.no_learn_legend))

    legend_ids = _find_legend_node_ids(g.nodes)
    if legend_ids:
        # Keep legend nodes for learning, but exclude them from export.
        g = DiagramGraph(
            nodes=[n for n in g.nodes if n.id not in legend_ids],
            edges=[e for e in g.edges if (e.src not in legend_ids and e.dst not in legend_ids)],
            mxgraphmodel_xml=g.mxgraphmodel_xml,
        )
        kind_of = {nid: k for nid, k in kind_of.items() if nid not in legend_ids}

    title = args.title if args.title else in_path.name
    md = graph_to_markdown(
        g,
        title=title,
        kind_of=kind_of,
        include_stats=(not args.no_stats),
        include_warnings=(not args.no_warnings),
        include_ids=(not args.no_ids),
        include_kind_tags=(not args.no_kind_tags),
        learned=learned,
    )

    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
    else:
        sys.stdout.write(md)

    if args.dump_json:
        Path(args.dump_json).write_text(
            json.dumps(graph_to_json_obj(g, kind_of), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
