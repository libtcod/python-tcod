#!/usr/bin/env python3
"""Print the description used for GitHub Releases."""
import re

TAG_BANNER = r"## \[[\w.]*\] - \d+-\d+-\d+\n"

RE_BODY = re.compile(fr".*?{TAG_BANNER}(.*?){TAG_BANNER}", re.DOTALL)


def main() -> None:
    """Output the most recently tagged changelog body to stdout."""
    with open("CHANGELOG.md", "r", encoding="utf-8") as f:
        match = RE_BODY.match(f.read())
    assert match
    body = match.groups()[0].strip()

    print(body)


if __name__ == "__main__":
    main()
