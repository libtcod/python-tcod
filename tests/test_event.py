"""Tests for event parsing and handling."""

from typing import Any, Final

import pytest

import tcod.event
import tcod.sdl.sys
from tcod._internal import _check
from tcod.cffi import ffi, lib
from tcod.event import KeySym, Modifier, Scancode

EXPECTED_EVENTS: Final = (
    tcod.event.Quit(),
    tcod.event.KeyDown(scancode=Scancode.A, sym=KeySym.A, mod=Modifier(0), pressed=True),
    tcod.event.KeyUp(scancode=Scancode.A, sym=KeySym.A, mod=Modifier(0), pressed=False),
)
"""Events to compare with after passing though the SDL event queue."""


def as_sdl_event(event: tcod.event.Event) -> dict[str, dict[str, Any]]:
    """Convert events into SDL_Event unions using cffi's union format."""
    match event:
        case tcod.event.Quit():
            return {"quit": {"type": lib.SDL_EVENT_QUIT}}
        case tcod.event.KeyboardEvent():
            return {
                "key": {
                    "type": (lib.SDL_EVENT_KEY_UP, lib.SDL_EVENT_KEY_DOWN)[event.pressed],
                    "scancode": event.scancode,
                    "key": event.sym,
                    "mod": event.mod,
                    "down": event.pressed,
                    "repeat": event.repeat,
                }
            }
    raise AssertionError


EVENT_PACK: Final = ffi.new("SDL_Event[]", [as_sdl_event(_e) for _e in EXPECTED_EVENTS])
"""A custom C array of SDL_Event unions based on EXPECTED_EVENTS."""


def push_events() -> None:
    """Reset the SDL event queue to an expected list of events."""
    tcod.sdl.sys.init(tcod.sdl.sys.Subsystem.EVENTS)  # Ensure SDL event queue is enabled

    lib.SDL_PumpEvents()  # Clear everything from the queue
    lib.SDL_FlushEvents(lib.SDL_EVENT_FIRST, lib.SDL_EVENT_LAST)

    assert _check(  # Fill the queue with EVENT_PACK
        lib.SDL_PeepEvents(EVENT_PACK, len(EVENT_PACK), lib.SDL_ADDEVENT, lib.SDL_EVENT_FIRST, lib.SDL_EVENT_LAST)
    ) == len(EVENT_PACK)


def test_get_events() -> None:
    push_events()
    assert tuple(tcod.event.get()) == EXPECTED_EVENTS

    assert tuple(tcod.event.get()) == ()
    assert tuple(tcod.event.wait(timeout=0)) == ()

    push_events()
    assert tuple(tcod.event.wait()) == EXPECTED_EVENTS


def test_event_dispatch() -> None:
    push_events()
    with pytest.deprecated_call():
        tcod.event.EventDispatch().event_wait(timeout=0)
    push_events()
    with pytest.deprecated_call():
        tcod.event.EventDispatch().event_get()
