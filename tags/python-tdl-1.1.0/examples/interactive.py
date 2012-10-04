#!/usr/bin/env python
"""
    Not much commentary in this example.  It's more of a demo.
"""
import sys
import code
import textwrap
import io

sys.path.insert(0, '../')
import tdl

sys.ps1 = '>>> '
sys.ps2 = '... '

WIDTH, HEIGHT = 80, 50
console = tdl.init(WIDTH, HEIGHT, 'Python Interpeter in TDL')

def scroll():
    console.blit(console, 0, 0, None, HEIGHT-1 , 0, 1)
    console.drawRect(0, HEIGHT-1, None, 1, ' ', (255, 255, 255), (0, 0, 0))

class TDLPrint(io.TextIOBase):
    def __init__(self, fgcolor=(255, 255, 255), bgcolor=(0, 0, 0)):
        self.colors = fgcolor, bgcolor

    def write(self, string):
        for strings in string.split('\n'):
            for string in textwrap.wrap(strings, 80):
                scroll()
                console.drawStr(0, HEIGHT-1, string, *self.colors)
                tdl.flush()

sys.stdout = TDLPrint()
olderr = sys.stderr
newerr = TDLPrint((255, 255, 255), (127, 0, 0))
interpeter = code.InteractiveConsole({'tdl':tdl, 'console':console})
print('Python %s' % sys.version)
print('Press ESC to quit')
if __name__ == '__main__':
    buffer = ''
    commands = ['']
    banner = sys.ps1
    cursor = 0
    scroll()
    while 1:
        console.drawRect(0, HEIGHT-1, None, 1, ' ', (255, 255, 255), (0, 0, 0))
        console.drawStr(0, HEIGHT-1, banner + buffer)
        try:
            console.drawChar(len(banner) + cursor, HEIGHT-1, None, None, (0, 255, 255))
        except tdl.TDLError:
            pass
        tdl.flush()
        
        for event in tdl.event.get():
            if event.type == 'QUIT':
                raise SystemExit()
            if event.type == 'KEYDOWN':
                if event.key == 'ENTER':
                    sys.stderr = newerr
                    try:
                        console.drawRect(0, HEIGHT-1, None, 1, None, (255, 255, 255), (0, 0, 0))
                        if interpeter.push(buffer):
                            banner = sys.ps2
                        else:
                            banner = sys.ps1
                        scroll()
                    except SystemExit:
                        raise
                    except:
                        sys.excepthook(*sys.exc_info())
                        banner = sys.ps1
                    sys.stderr = olderr
                    if buffer not in commands:
                        commands.append(buffer)
                    buffer = ''
                elif event.key == 'BACKSPACE':
                    if cursor == 0:
                        continue
                    if buffer[:cursor][-4:] == '    ':
                        buffer = buffer[:cursor-4] + buffer[cursor:]
                        cursor -= 4
                    elif buffer:
                        buffer = buffer[:cursor-1] + buffer[cursor:]
                        cursor -= 1
                elif event.key == 'DELETE':
                    buffer = buffer[:cursor] + buffer[cursor+1:]
                elif event.key == 'LEFT':
                    cursor -= 1
                elif event.key == 'RIGHT':
                    cursor += 1
                elif event.key == 'HOME':
                    cursor = 0
                elif event.key == 'END':
                    cursor = len(buffer)
                elif event.key == 'UP':
                    commands.insert(0, buffer)
                    buffer = commands.pop()
                    cursor = len(buffer)
                elif event.key == 'DOWN':
                    commands.append(buffer)
                    buffer = commands.pop(0)
                    cursor = len(buffer)
                elif event.key == 'TAB':
                    buffer = buffer[:cursor] + '    ' + buffer[cursor:]
                    cursor += 4
                elif event.key == 'ESCAPE':
                    raise SystemExit()
                elif event.char:
                    buffer = buffer[:cursor] + event.char + buffer[cursor:]
                    cursor += 1
                cursor = max(0, min(cursor, len(buffer)))
