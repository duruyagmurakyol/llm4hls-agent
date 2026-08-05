"""Small source-text helpers used by structural HLS checks."""

from __future__ import annotations


def mask_cpp_comments(source: str) -> str:
    """Replace C/C++ comment contents with spaces while preserving positions.

    Newlines and the total source length are preserved so regex matches and loop
    spans found in the masked text still map directly onto the original source.
    String and character literals are left untouched, including comment-like
    text inside them.
    """
    masked = list(source)
    index = 0
    state = "code"

    while index < len(source):
        current = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""

        if state == "code":
            if current == "/" and following == "/":
                masked[index] = " "
                masked[index + 1] = " "
                index += 2
                state = "line_comment"
                continue
            if current == "/" and following == "*":
                masked[index] = " "
                masked[index + 1] = " "
                index += 2
                state = "block_comment"
                continue
            if current == '"':
                state = "string"
            elif current == "'":
                state = "character"
            index += 1
            continue

        if state == "line_comment":
            if current == "\n":
                state = "code"
            else:
                masked[index] = " "
            index += 1
            continue

        if state == "block_comment":
            if current == "*" and following == "/":
                masked[index] = " "
                masked[index + 1] = " "
                index += 2
                state = "code"
                continue
            if current != "\n":
                masked[index] = " "
            index += 1
            continue

        if state in {"string", "character"}:
            if current == "\\" and index + 1 < len(source):
                index += 2
                continue
            if state == "string" and current == '"':
                state = "code"
            elif state == "character" and current == "'":
                state = "code"
            index += 1
            continue

    return "".join(masked)
