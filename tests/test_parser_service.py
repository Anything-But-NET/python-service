import pytest

from python_service.services.parse_service import MarkdownParserService


@pytest.fixture
def parser():
    return MarkdownParserService()


def test_parse_preamble_only(parser):
    text = "Line 1\nLine 2\nLine 3"
    root = parser.parse(text)
    assert root.name == "Root"
    assert root.level == 0
    assert len(root.children) == 1
    preamble = root.children[0]
    assert preamble.name == "Preamble"
    assert preamble.content == "Line 1\nLine 2\nLine 3"


def test_parse_preamble_and_headers(parser):
    text = """Intro text here.
# Main Title
Section content.
## Subsection 1
Sub 1 text.
### Deep Section
Deep text.
## Subsection 2
Sub 2 text.
# Final Title
Final text.
"""
    root = parser.parse(text)
    assert len(root.children) == 3
    preamble, main_title, final_title = root.children

    assert preamble.name == "Preamble"
    assert preamble.content == "Intro text here."

    assert main_title.name == "Main Title"
    assert main_title.content == "Section content."
    assert len(main_title.children) == 2

    sub1, sub2 = main_title.children
    assert sub1.name == "Subsection 1"
    assert sub1.content == "Sub 1 text."
    assert len(sub1.children) == 1
    assert sub1.children[0].name == "Deep Section"
    assert sub1.children[0].content == "Deep text."

    assert sub2.name == "Subsection 2"
    assert sub2.content == "Sub 2 text."

    assert final_title.name == "Final Title"
    assert final_title.content == "Final text."


def test_parse_code_blocks_ignore_inner_headers(parser):
    text = """# Main
```python
# This is a comment, not a markdown header
def hello():
    return 1
```
End of code.
~~~markdown
## Another ignored header inside tilde block
~~~
After tildes.
"""
    root = parser.parse(text)
    assert len(root.children) == 1
    main_node = root.children[0]
    assert main_node.name == "Main"
    assert len(main_node.children) == 0
    assert "# This is a comment, not a markdown header" in main_node.content
    assert "## Another ignored header inside tilde block" in main_node.content
    assert "After tildes." in main_node.content


def test_parse_empty_string(parser):
    root = parser.parse("")
    assert root.name == "Root"
    assert len(root.children) == 0
    assert root.content == ""
