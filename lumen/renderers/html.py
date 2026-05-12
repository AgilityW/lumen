"""WeChat Official Account HTML renderer — inline CSS, no JS, mobile-first.

Conforms to WeChat's article editor constraints:
- Inline CSS only (no <style> or <link>)
- No JavaScript
- Mobile-optimized single-column layout
- Clean typography for Chinese + English mixed content
"""

from datetime import datetime
from typing import Any


def _escape(text: Any) -> str:
    """HTML-escape a value for safe inline output."""
    s = str(text)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _tag(
    name: str,
    content: str = "",
    attrs: dict[str, str] | None = None,
    inline: bool = False,
) -> str:
    """Build an HTML tag with inline styles in the attrs dict."""
    attr_str = ""
    if attrs:
        attr_str = " " + " ".join(f'{k}="{_escape(v)}"' for k, v in attrs.items())
    if inline:
        return f"<{name}{attr_str}>{content}</{name}>"
    return f"<{name}{attr_str}>\n{content}\n</{name}>"


class WeChatRenderer:
    """Render book synthesis as WeChat Official Account-compatible HTML."""

    def __init__(self, book_title: str = ""):
        self.book_title = book_title

    def render(self, synthesis: dict) -> str:
        """Generate full HTML document ready for WeChat article editor."""
        summary = synthesis.get("book_summary", "")
        concepts = synthesis.get("core_concepts", [])
        arguments = synthesis.get("key_arguments", [])
        rel_map = synthesis.get("relationship_map", [])
        reading_notes = synthesis.get("reading_notes", [])
        date = datetime.now().strftime("%Y-%m-%d")

        sections: list[str] = []

        # Title
        sections.append(self._render_header(synthesis))

        # Summary
        if summary:
            sections.append(self._render_summary(summary))

        # Core Concepts
        if concepts:
            sections.append(self._render_concepts(concepts))

        # Key Arguments
        if arguments:
            sections.append(self._render_arguments(arguments))

        # Relationship Map
        if rel_map:
            sections.append(self._render_relationships(rel_map))

        # Reading Notes
        if reading_notes:
            sections.append(self._render_reading_notes(reading_notes))

        # Footer
        sections.append(self._render_footer(date))

        body = "\n\n".join(sections)

        css = (
            "max-width:677px;margin:0 auto;padding:10px 16px 30px;"
            "font-family:-apple-system,BlinkMacSystemFont,"
            '"Helvetica Neue","PingFang SC","Microsoft YaHei",sans-serif;'
            "font-size:16px;line-height:1.8;color:#333;word-wrap:break-word"
        )
        return f'<section style="{_escape(css)}">\n{body}\n</section>'

    def _render_header(self, synthesis: dict) -> str:
        title = synthesis.get("title", self.book_title) or "Book Notes"
        summary = synthesis.get("book_summary", "")
        excerpt = summary[:120] + "…" if len(summary) > 120 else summary

        lines = [
            _tag("h2", _escape(title), {
                "style": "font-size:20px;font-weight:700;margin:0 0 8px;line-height:1.4;color:#111;text-align:center",
            }),
        ]
        if excerpt:
            lines.append(
                _tag("p", _escape(excerpt), {
                    "style": "font-size:14px;color:#888;margin:0 0 4px;text-align:center",
                })
            )
        lines.append(
            _tag("hr", "", {
                "style": "border:none;border-top:1px solid #e0e0e0;margin:20px 0",
            })
        )
        return "\n".join(lines)

    def _render_summary(self, summary: str) -> str:
        paragraphs = summary.strip().split("\n")
        content = "\n".join(
            _tag("p", _escape(p), {
                "style": "margin:0 0 12px;text-indent:2em",
            })
            for p in paragraphs if p.strip()
        )
        return content

    def _render_concepts(self, concepts: list[dict]) -> str:
        items: list[str] = []
        items.append(
            _tag("h3", "📌 核心概念", {
                "style": "font-size:18px;font-weight:700;margin:24px 0 12px;color:#111;"
                         "padding-bottom:6px;border-bottom:2px solid #4a90d9",
            })
        )
        for c in concepts:
            name = c.get("name", "").strip()
            definition = c.get("definition", "").strip()
            importance = c.get("importance", "").strip()
            if not name:
                continue

            parts = [_tag("strong", _escape(name), {
                "style": "color:#4a90d9;font-size:16px",
            })]
            if definition:
                parts.append(f"：{_escape(definition)}")
            if importance:
                parts.append(
                    _tag("span", f" [{_escape(importance)}]", {
                        "style": "color:#999;font-size:13px",
                    })
                )
            items.append(
                _tag("p", "".join(parts), {
                    "style": "margin:8px 0;padding-left:12px;border-left:3px solid #4a90d9",
                })
            )
        return "\n".join(items)

    def _render_arguments(self, arguments: list[str]) -> str:
        items: list[str] = [
            _tag("h3", "💡 关键论点", {
                "style": "font-size:18px;font-weight:700;margin:24px 0 12px;color:#111;"
                         "padding-bottom:6px;border-bottom:2px solid #e8a838",
            })
        ]
        for arg in arguments:
            items.append(
                _tag("li", _escape(arg), {
                    "style": "margin:0 0 8px;padding-left:4px;color:#444",
                })
            )
        return _tag("ul", "\n".join(items), {
            "style": "padding-left:20px;list-style:disc",
        })

    def _render_relationships(self, rel_map: list[dict]) -> str:
        items: list[str] = [
            _tag("h3", "🔗 关系图谱", {
                "style": "font-size:18px;font-weight:700;margin:24px 0 12px;color:#111;"
                         "padding-bottom:6px;border-bottom:2px solid #6bbf6b",
            })
        ]
        for rel in rel_map:
            from_ = rel.get("from", "?")
            to_ = rel.get("to", "?")
            rtype = rel.get("type", "relates-to")
            items.append(
                _tag("li", f"{_escape(from_)} ——{_escape(rtype)}——→ {_escape(to_)}", {
                    "style": "margin:0 0 6px;padding-left:4px;color:#444",
                })
            )
        return _tag("ul", "\n".join(items), {
            "style": "padding-left:20px;list-style:none",
        })

    def _render_reading_notes(self, notes: list | str) -> str:
        items: list[str] = [
            _tag("h3", "📝 阅读笔记", {
                "style": "font-size:18px;font-weight:700;margin:24px 0 12px;color:#111;"
                         "padding-bottom:6px;border-bottom:2px solid #ab8ed4",
            })
        ]
        notes_list = notes if isinstance(notes, list) else [notes]
        for note in notes_list:
            if isinstance(note, dict):
                text = note.get("note", note.get("text", str(note)))
            else:
                text = str(note)
            items.append(
                _tag("blockquote", _escape(text), {
                    "style": "margin:8px 0;padding:10px 14px;background:#f7f7f7;"
                             "border-left:4px solid #ab8ed4;color:#555;font-size:15px",
                })
            )
        return "\n".join(items)

    def _render_footer(self, date: str) -> str:
        return "\n".join([
            _tag("hr", "", {
                "style": "border:none;border-top:1px solid #e0e0e0;margin:30px 0 16px",
            }),
            _tag("p", f"Generated by Lumen · {_escape(date)}", {
                "style": "font-size:13px;color:#bbb;text-align:center;margin:0",
            }),
        ])
