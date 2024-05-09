"""Hello world using tcod's SDL API and using Pillow for the TTF rendering."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont  # pip install Pillow

import tcod.event
import tcod.sdl.render
import tcod.sdl.video

CURRENT_DIR = Path(__file__).parent  # Directory of this script.
font = ImageFont.truetype(bytes(CURRENT_DIR / "DejaVuSerif.ttf"), size=18)  # Preloaded font file.


def render_text(renderer: tcod.sdl.render.Renderer, text: str) -> tcod.sdl.render.Texture:
    """Render text, upload it to VRAM, then return it as an SDL Texture."""
    # Use Pillow to render the font.
    _left, _top, right, bottom = font.getbbox(text)
    width, height = right, bottom
    image = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font)
    # Push to VRAM using SDL.
    texture = renderer.upload_texture(np.asarray(image))
    texture.blend_mode = tcod.sdl.render.BlendMode.BLEND  # Enable alpha blending by default.
    return texture


def main() -> None:
    """Show hello world until the window is closed."""
    # Open an SDL window and renderer.
    window = tcod.sdl.video.new_window(720, 480, flags=tcod.sdl.video.WindowFlags.RESIZABLE)
    renderer = tcod.sdl.render.new_renderer(window, target_textures=True)
    # Render the text once, then reuse the texture.
    hello_world = render_text(renderer, "Hello World")
    hello_world.color_mod = (64, 255, 64)  # Set the color when copied.

    while True:
        renderer.draw_color = (0, 0, 0, 255)
        renderer.clear()
        renderer.copy(hello_world, dest=(0, 0, hello_world.width, hello_world.height))
        renderer.present()
        for event in tcod.event.get():
            if isinstance(event, tcod.event.Quit):
                raise SystemExit()


if __name__ == "__main__":
    main()
