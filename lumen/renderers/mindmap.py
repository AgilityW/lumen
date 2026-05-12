"""Mermaid mind map renderer.

Uses graph LR for relationship-rich diagrams (supports arrows).
Node styling: [[name]] for concepts (rectangle with rounded edges),
[label] for arguments (stadium shape), plain text for groups.
"""

from typing import Any


class MindmapRenderer:
    """Generate Mermaid graph syntax from book synthesis."""

    def render(self, synthesis: dict) -> str:
        concepts = synthesis.get("core_concepts", [])
        arguments = synthesis.get("key_arguments", [])
        rel_map = synthesis.get("relationship_map", [])

        lines = ["flowchart LR"]
        lines.append("  B((Book))")

        node_id = 0
        concept_ids = {}

        # ── Concepts section ──
        if concepts:
            lines.append("  subgraph Concepts[Core Concepts]")
            _concept_ids = []
            for c in concepts[:15]:
                node_id += 1
                cid = f"C{node_id}"
                name = c.get("name", "?").replace('"', "'")
                lines.append(f"    {cid}[[{name}]]")
                _concept_ids.append(cid)
                concept_ids[name] = cid
            lines.append("  end")
            lines.append("  B --> Concepts")

        # ── Arguments section ──
        if arguments:
            lines.append("  subgraph Arguments[Key Arguments]")
            for a in arguments[:10]:
                node_id += 1
                aid = f"A{node_id}"
                label = a[:60].replace('"', "'")
                if len(a) > 60:
                    label += "..."
                lines.append(f"    {aid}[{label}]")
            lines.append("  end")
            lines.append("  B --> Arguments")

        # ── Relationship section ──
        if rel_map:
            lines.append("  subgraph Relationships[Relationships]")
            seen = set()
            for r in rel_map[:10]:
                from_ = r.get("from", "?")
                to_ = r.get("to", "?")
                rtype = r.get("type", "relates")
                key = (from_, rtype, to_)
                if key not in seen:
                    seen.add(key)
                    # Look up concept node IDs, fall back to inline labels
                    src = concept_ids.get(from_, f'"{from_}"')
                    dst = concept_ids.get(to_, f'"{to_}"')
                    lines.append(f"    {src} -- {rtype} --> {dst}")
            lines.append("  end")
            lines.append("  B --> Relationships")

        return "\n".join(lines)
