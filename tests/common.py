
import pytest

import tcod


def raise_Exception(*args):
    raise Exception('testing exception')

needs_window = pytest.mark.skipif(
    pytest.config.getoption("--no-window"),
    reason="This test needs a rendering context."
)