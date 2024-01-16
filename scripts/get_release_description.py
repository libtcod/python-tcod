#!/usr/bin/env python
"""Print the description used for GitHub Releases."""
from __future__ import annotations

import re
from pathlib import Path

TAG_BANNER = r"## \[[\w.]*\] - \d+-\d+-\d+\n"

RE_BODY = re.compile(rf".*?{TAG_BANNER}(.*?){TAG_BANNER}", re.DOTALL)


def main() -> None:
    """Output the most recently tagged changelog body to stdout."""
    match = RE_BODY.match(Path("CHANGELOG.md").read_text(encoding="utf-8"))
    assert match
    body = match.groups()[0].strip()

    print(body)


if __name__ == "__main__":
    main()
