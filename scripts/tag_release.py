#!/usr/bin/env python
"""Automate tagged releases of this project."""

from __future__ import annotations

import argparse
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

# ruff: noqa: S603, S607

PROJECT_DIR = Path(__file__).parent.parent

parser = argparse.ArgumentParser(description="Tags and releases the next version of this project.")

parser.add_argument("tag", help="Semantic version number to use as the tag.")

parser.add_argument("-e", "--edit", action="store_true", help="Force edits of git commits.")

parser.add_argument("-n", "--dry-run", action="store_true", help="Don't modify files.")

parser.add_argument("-v", "--verbose", action="store_true", help="Print debug information.")


def parse_changelog(args: argparse.Namespace) -> tuple[str, str]:
    """Return an updated changelog and and the list of changes."""
    match = re.match(
        pattern=r"(.*?## \[Unreleased]\n)(.+?\n)(\n*## \[.*)",
        string=(PROJECT_DIR / "CHANGELOG.md").read_text(encoding="utf-8"),
        flags=re.DOTALL,
    )
    assert match
    header, changes, tail = match.groups()

    iso_date = datetime.datetime.now(tz=datetime.timezone.utc).date().isoformat()
    tagged = f"\n## [{args.tag}] - {iso_date}\n{changes}"
    if args.verbose:
        print("--- Tagged section:")
        print(tagged)

    return f"{header}{tagged}{tail}", changes


def replace_unreleased_tags(tag: str, *, dry_run: bool) -> None:
    """Walk though sources and replace pending tags with the new tag."""
    match = re.match(r"\d+\.\d+", tag)
    assert match
    short_tag = match.group()
    for directory, _, files in os.walk(PROJECT_DIR / "tcod"):
        for filename in files:
            file = Path(directory, filename)
            if file.suffix != ".py":
                continue
            text = file.read_text(encoding="utf-8")
            new_text = re.sub(r":: *unreleased", rf":: {short_tag}", text, flags=re.IGNORECASE)
            if text != new_text:
                print(f"Update tags in {file}")
                if not dry_run:
                    file.write_text(new_text, encoding="utf-8")


def main() -> None:
    """Entry function."""
    if len(sys.argv) <= 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    new_changelog, changes = parse_changelog(args)

    if args.verbose:
        print("--- New changelog:")
        print(new_changelog)

    replace_unreleased_tags(args.tag, dry_run=args.dry_run)

    if not args.dry_run:
        (PROJECT_DIR / "CHANGELOG.md").write_text(new_changelog, encoding="utf-8")
        edit = ["-e"] if args.edit else []
        subprocess.check_call(["git", "commit", "-avm", f"Prepare {args.tag} release.", *edit])
        subprocess.check_call(["git", "tag", args.tag, "-am", f"{args.tag}\n\n{changes}", *edit])


if __name__ == "__main__":
    main()
