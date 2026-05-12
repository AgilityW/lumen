import pytest

from lumen.core.classifier import classify, get_framework_name


class TestClassifier:
    def test_classify_book(self):
        text = (
            "Chapter 1: Introduction\n"
            "Some content here...\n"
            "Chapter 2: Advanced Topics\n"
            "More content...\n"
            "Chapter 3: Deeper Dive\n"
            "Even more...\n"
            "Chapter 4: Practical Examples\n"
            "Examples...\n"
            "Chapter 5: Conclusion\n"
            "Final words.\n"
            "# Table of Contents\n"
            "ISBN: 978-0-123456-78-9"
        )
        result = classify(text)
        assert result == "book"

    def test_classify_podcast_with_timecodes(self):
        text = (
            "[00:00] Host: Welcome to the show\n"
            "[05:30] Guest: Thanks for having me\n"
            "[12:15] Guest: Let me explain the key insight\n"
            "[20:00] Host: That's interesting\n"
            "[30:45] Guest: There's another angle here\n"
            "[40:00] Host: Let's wrap up\n"
        )
        result = classify(text)
        assert result == "podcast"

    def test_classify_article(self):
        text = (
            "# Abstract\n"
            "This paper presents a novel approach...\n"
            "# Motivation\n"
            "Why this matters...\n"
            "# Background\n"
            "Related work...\n"
            "# Methodology\n"
            "Our approach...\n"
            "# Conclusion\n"
            "Summary of findings...\n"
            "# References\n"
            "[1] Smith et al."
        )
        result = classify(text)
        assert result == "article"

    def test_classify_unknown(self):
        text = "Just some random text without any clear signals."
        result = classify(text)
        assert result == "unknown"

    def test_empty_text_returns_unknown(self):
        assert classify("") == "unknown"

    def test_book_chapter_density_boost(self):
        text = (
            "Chapter 1: Start\nsome text\n"
            "Chapter 2: Middle\nmore text\n"
            "Chapter 3: End\nfinal text\n"
            "ISBN: 978-0-123456-78-9\n"
            "# Table of Contents\n"
            "Introduction\nsome preface\n"
        )
        result = classify(text)
        assert result == "book"


class TestGetFrameworkName:
    def test_podcast_maps_to_podcast(self):
        assert get_framework_name("podcast") == "podcast"

    def test_book_maps_to_book(self):
        assert get_framework_name("book") == "book"

    def test_unknown_maps_to_default(self):
        assert get_framework_name("unknown") == "default"

    def test_unrecognized_maps_to_default(self):
        assert get_framework_name("something_else") == "default"
