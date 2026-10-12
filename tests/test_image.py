"""Tests for the image object and its NumPy interface."""

import numpy as np
import pytest

import tcod.image


@pytest.mark.parametrize("shape", [(4, 6), (6, 4), (4, 4)])
@pytest.mark.parametrize("rotations", [-5, -2, -1, 0, 1, 2, 3, 4, 5])
def test_rotate90_dimensions(shape: tuple[int, int], rotations: int) -> None:
    pixels = np.arange(shape[0] * shape[1] * 3, dtype=np.uint8).reshape(*shape, 3)
    image = tcod.image.Image.from_array(pixels)
    image.rotate90(rotations)
    expected = np.rot90(pixels, -rotations)
    assert (image.height, image.width) == expected.shape[:2]
    np.testing.assert_array_equal(np.asarray(image), expected)


def test_rotate90_sequence() -> None:
    pixels = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    image = tcod.image.Image.from_array(pixels)
    for _ in range(4):
        image.rotate90()
        pixels = np.rot90(pixels, -1)
        assert (image.height, image.width) == pixels.shape[:2]
        np.testing.assert_array_equal(np.asarray(image), pixels)
