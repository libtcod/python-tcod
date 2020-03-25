To use these BDF Unicode fonts with python-tcod you should use the
`tcod.tileset` module:

```python
import tcod
import tcod.tileset

# Load the BDF file as the active tileset.
tcod.tileset.set_default(tcod.tileset.load_bdf("file_to_load.bdf"))

# Start python-tcod normally.
with tcod.console_init_root(...) as root_console:
    ...
```
