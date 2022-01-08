#!/usr/bin/env python3
import argparse
import datetime
import re
import subprocess
import sys
from typing import Tuple

parser = argparse.ArgumentParser(description="Tags and releases the next version of this project.")

parser.add_argument("tag", help="Semantic version number to use as the tag.")

parser.add_argument("-e", "--edit", action="store_true", help="Force edits of git commits.")

parser.add_argument("-n", "--dry-run", action="store_true", help="Don't modify files.")

parser.add_argument("-v", "--verbose", action="store_true", help="Print debug information.")


def parse_changelog(args: argparse.Namespace) -> Tuple[str, str]:
    """Return an updated changelog and and the list of changes."""
    with open("CHANGELOG.md", "r", encoding="utf-8") as file:
        match = re.match(
            pattern=r"(.*?## \[Unreleased]\n)(.+?\n)(\n*## \[.*)",
            string=file.read(),
            flags=re.DOTALL,
        )
    assert match
    header, changes, tail = match.groups()
    tagged = "\n## [%s] - %s\n%s" % (
        args.tag,
        datetime.date.today().isoformat(),
        changes,
    )
    if args.verbose:
        print("--- Tagged section:")
        print(tagged)

    return "".join((header, tagged, tail)), changes


def main() -> None:
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    new_changelog, changes = parse_changelog(args)

    if args.verbose:
        print("--- New changelog:")
        print(new_changelog)

    if not args.dry_run:
        with open("CHANGELOG.md", "w", encoding="utf-8") as f:
            f.write(new_changelog)
        edit = ["-e"] if args.edit else []
        subprocess.check_call(["git", "commit", "-avm", "Prepare %s release." % args.tag] + edit)
        subprocess.check_call(["git", "tag", args.tag, "-am", "%s\n\n%s" % (args.tag, changes)] + edit)


if __name__ == "__main__":
    main()
