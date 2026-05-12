"""Obsidian note renderer — writes structured book notes to an Obsidian vault."""

from datetime import datetime
from pathlib import Path


class ObsidianRenderer:
    """Write book synthesis to vault as structured Obsidian markdown notes."""

    def __init__(self, vault_path: str, book_dir: str = "Books"):
        self.vault_path = vault_path
        self.book_dir = book_dir

    def render(self, book_slug: str, synthesis: dict) -> str:
        """Write book note + concept notes to vault. Returns the note file path."""
        vault = Path(self.vault_path)
        book_path = vault / self.book_dir / book_slug
        book_path.mkdir(parents=True, exist_ok=True)

        # Main book note
        main_path = book_path / f"{book_slug}.md"
        content = self._render_book_note(book_slug, synthesis)
        main_path.write_text(content, encoding="utf-8")

        # Individual concept notes (with dedup by slug)
        concepts = synthesis.get("core_concepts", [])
        seen_slugs = set()
        for concept in concepts:
            cname = concept.get("name", "").strip()
            if not cname:
                continue
            slug = cname.lower().replace(" ", "-").replace("/", "-")
            slug = slug.replace("--", "-").strip("-")
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            note_path = book_path / f"{slug}.md"
            note_content = self._render_concept_note(cname, concept, book_slug, synthesis)
            note_path.write_text(note_content, encoding="utf-8")

        return str(main_path)

    def write_mindmap(self, book_slug: str, mermaid_content: str) -> None:
        """Write Mermaid mind map as an embedded markdown block in the book note."""
        vault = Path(self.vault_path)
        book_path = vault / self.book_dir / book_slug
        main_path = book_path / f"{book_slug}.md"

        if not main_path.exists():
            return

        existing = main_path.read_text(encoding="utf-8")

        mm_section = "\n\n## Mind Map\n\n```mermaid\n" + mermaid_content + "\n```\n"
        if "## Mind Map" not in existing:
            main_path.write_text(existing + mm_section, encoding="utf-8")
        else:
            # Replace existing mind map section
            import re
            updated = re.sub(
                r"\n## Mind Map\n\n```mermaid\n.*?\n```\n?",
                mm_section,
                existing,
                flags=re.DOTALL,
            )
            main_path.write_text(updated, encoding="utf-8")

    def _render_book_note(self, book_slug: str, synthesis: dict) -> str:
        """Render the main book note with frontmatter, summary, and concept links."""
        date = datetime.now().strftime("%Y-%m-%d")
        summary = synthesis.get("book_summary", "No summary available.")
        concepts = synthesis.get("core_concepts", [])
        arguments = synthesis.get("key_arguments", [])
        relationship_map = synthesis.get("relationship_map", [])
        reading_notes = synthesis.get("reading_notes", [])

        lines = [
            "---",
            f"title: \"{book_slug}\"",
            f"date: {date}",
            "tags: [book, lumen]",
            "---",
            "",
            f"# {book_slug}",
            "",
            "## Summary",
            "",
            summary,
            "",
        ]

        if concepts:
            lines.append("## Core Concepts")
            lines.append("")
            for c in concepts:
                cname = c.get("name", "?")
                cslug = cname.lower().replace(" ", "-").replace("/", "-")
                cdef = c.get("definition", "")
                importance = c.get("importance", "")
                lines.append(f"- **[[{cslug}]]**: {cdef}" + (f" _{importance}_" if importance else ""))
            lines.append("")

        if arguments:
            lines.append("## Key Arguments")
            lines.append("")
            for arg in arguments:
                lines.append(f"- {arg}")
            lines.append("")

        if relationship_map:
            lines.append("## Relationship Map")
            lines.append("")
            for rel in relationship_map:
                from_ = rel.get("from", "?")
                to_ = rel.get("to", "?")
                rtype = rel.get("type", "relates-to")
                lines.append(f"- **{from_}** _{rtype}_ **{to_}**")
            lines.append("")

        if reading_notes:
            lines.append("## Reading Notes")
            lines.append("")
            notes_list = reading_notes if isinstance(reading_notes, list) else [reading_notes]
            for note in notes_list:
                if isinstance(note, str):
                    lines.append(f"- {note}")
                elif isinstance(note, dict):
                    lines.append(f"- {note.get('note', note.get('text', str(note)))}")
                else:
                    lines.append(f"- {str(note)}")
            lines.append("")

        return "\n".join(lines)

    def _render_concept_note(
        self, name: str, concept: dict, book_slug: str, synthesis: dict
    ) -> str:
        """Render an individual concept note with backlinks to the book."""
        date = datetime.now().strftime("%Y-%m-%d")
        definition = concept.get("definition", "")
        importance = concept.get("importance", "")

        lines = [
            "---",
            f"title: \"{name}\"",
            f"date: {date}",
            f"tags: [concept, book/{book_slug}]",
            "---",
            "",
            f"# {name}",
            "",
        ]
        if definition:
            lines.append(definition)
            lines.append("")
        if importance:
            lines.append(f"_{importance}_")
            lines.append("")

        lines.append("---")
        lines.append(f"Sourced from: **[[{book_slug}]]**")
        lines.append("")

        return "\n".join(lines)
