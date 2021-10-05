#!/usr/bin/env python3
import argparse
import datetime
import re
import subprocess
import sys
from typing import Any, Tuple

parser = argparse.ArgumentParser(description="Tags and releases the next version of this project.")

parser.add_argument("tag", help="Semantic version number to use as the tag.")

parser.add_argument("-e", "--edit", action="store_true", help="Force edits of git commits.")

parser.add_argument("-n", "--dry-run", action="store_true", help="Don't modify files.")

parser.add_argument("-v", "--verbose", action="store_true", help="Print debug information.")


def parse_changelog(args: Any) -> Tuple[str, str]:
    """Return an updated changelog and and the list of changes."""
    with open("CHANGELOG.rst", "r", encoding="utf-8") as file:
        match = re.match(
            pattern=r"(.*?Unreleased\n---+\n)(.+?)(\n*[^\n]+\n---+\n.*)",
            string=file.read(),
            flags=re.DOTALL,
        )
    assert match
    header, changes, tail = match.groups()
    tag = "%s - %s" % (args.tag, datetime.date.today().isoformat())

    tagged = "\n%s\n%s\n%s" % (tag, "-" * len(tag), changes)
    if args.verbose:
        print(tagged)

    return "".join((header, tagged, tail)), changes


def main() -> None:
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    if args.verbose:
        print(args)

    new_changelog, changes = parse_changelog(args)

    if not args.dry_run:
        with open("CHANGELOG.rst", "w", encoding="utf-8") as f:
            f.write(new_changelog)
        edit = ["-e"] if args.edit else []
        subprocess.check_call(["git", "commit", "-avm", "Prepare %s release." % args.tag] + edit)
        subprocess.check_call(["git", "tag", args.tag, "-am", "%s\n\n%s" % (args.tag, changes)] + edit)


if __name__ == "__main__":
    main()
