#!/usr/bin/env python
"""This script is used to generate the tables for `charmap-reference.rst`.

Uses the tabulate module from PyPI.
"""

from __future__ import annotations

import argparse
import unicodedata
from typing import Iterable, Iterator

from tabulate import tabulate

import tcod.tileset


def get_character_maps() -> Iterator[str]:
    """Return an iterator of the current character maps from tcod.tileset."""
    for name in dir(tcod.tileset):
        if name.startswith("CHARMAP_"):
            yield name[len("CHARMAP_") :].lower()


def escape_rst(string: str) -> str:
    """Escape RST symbols and disable Sphinx smart quotes."""
    return (
        string.replace("\\", "\\\\")
        .replace("*", "\\*")
        .replace("|", "\\|")
        .replace("`", "\\`")
        .replace("'", "\\'")
        .replace('"', '\\"')
    )


def generate_table(charmap: Iterable[int]) -> str:
    """Generate and RST table for `charmap`."""
    headers = ("Tile Index", "Unicode", "String", "Name")
    table = []

    for i, ch in enumerate(charmap):
        hex_len = len(f"{ch:x}")
        if hex_len % 2:  # Prevent an odd number of hex digits.
            hex_len += 1
        try:
            name = unicodedata.name(chr(ch))
        except ValueError:
            # Skip names rather than guessing, the official names are enough.
            name = ""
        string = escape_rst(f"{chr(ch)!r}")
        table.append((i, f"0x{ch:0{hex_len}X}", string, name))
    return tabulate(table, headers, tablefmt="rst")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate an RST table for a tcod character map.",
    )
    parser.add_argument(
        "charmap",
        action="store",
        choices=list(get_character_maps()),
        type=str,
        help="which character map to generate a table from",
    )
    parser.add_argument(
        "-o",
        "--out-file",
        action="store",
        type=argparse.FileType("w", encoding="utf-8"),
        default="-",
        help="where to write the table to (stdout by default)",
    )
    args = parser.parse_args()
    charmap = getattr(tcod.tileset, f"CHARMAP_{args.charmap.upper()}")
    output = generate_table(charmap)
    with args.out_file as f:
        f.write(output)


if __name__ == "__main__":
    main()
