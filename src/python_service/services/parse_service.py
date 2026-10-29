import re

from python_service.models.domain import SegmentNode

HEADER_PATTERN = re.compile(r"^[ ]{0,3}(#{1,6})\s+(.+?)(?:\s+#+)?$")
FENCE_PATTERN = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})(.*)$")


class MarkdownParserService:
    def parse(self, md_text: str) -> SegmentNode:
        root = SegmentNode("Root", 0)
        stack = [root]

        preamble_node = SegmentNode("Preamble", 1)
        first_header_seen = False

        in_code_block = False
        fence_char = None
        fence_len = 0

        for line in md_text.splitlines():
            is_header = False

            if in_code_block:
                fence_match = FENCE_PATTERN.match(line)
                if (
                    fence_match
                    and fence_match.group(1)[0] == fence_char
                    and len(fence_match.group(1)) >= fence_len
                ):
                    in_code_block = False
            else:
                fence_match = FENCE_PATTERN.match(line)
                if fence_match:
                    in_code_block = True
                    fence_char = fence_match.group(1)[0]
                    fence_len = len(fence_match.group(1))
                else:
                    header_match = HEADER_PATTERN.match(line)
                    if header_match:
                        is_header = True

                        if not first_header_seen:
                            first_header_seen = True
                            if preamble_node.content.strip():
                                root.children.append(preamble_node)

                        level = len(header_match.group(1))
                        title = header_match.group(2).strip()
                        new_node = SegmentNode(title, level)

                        while stack and stack[-1].level >= level:
                            stack.pop()

                        stack[-1].children.append(new_node)
                        stack.append(new_node)

            if not is_header:
                if not first_header_seen:
                    preamble_node.add_content(line)
                else:
                    stack[-1].add_content(line)

        if not first_header_seen and preamble_node.content.strip():
            root.children.append(preamble_node)

        self._clean_tree(root)
        return root

    def _clean_tree(self, node: SegmentNode) -> None:
        node.content = node.content.strip()
        for child in node.children:
            self._clean_tree(child)
