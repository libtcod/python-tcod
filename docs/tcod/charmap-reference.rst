Character Table Reference
=========================

This document exists as an easy reference for using non-ASCII glyphs in
standard tcod functions.

*Tile Index* is the position of the glyph in the tileset image.
This is relevant for loading the tileset and for using :any:`Tileset.remap` to
reassign tiles to new code points.

*Unicode* is the Unicode code point as a hexadecimal number.
You can use :any:`chr` to convert these numbers into a string.
Character maps such as :any:`tcod.tileset.CHARMAP_CP437` are simply a list of
Unicode numbers, where the index of the list is the *Tile Index*.

*String* is the Python string for that character.
This lets you use that character inline with print functions.
These will work with :any:`ord` to convert them into a number.

*Name* is the official name of a Unicode character.

The symbols currently shown under *String* are provided by your browser, they
typically won't match the graphics provided by your tileset or could even be
missing from your browsers font entirely.  You could experience similar issues
with your editor and IDE.


.. _code-page-437:

Code Page 437
-------------

The layout for tilesets loaded with: :any:`tcod.tileset.CHARMAP_CP437`

This is one of the more common character mappings.
Used for several games in the DOS era, and still used today whenever you want
an old school aesthetic.

The Dwarf Fortress community is known to have a large collection of tilesets in
this format:
https://dwarffortresswiki.org/index.php/Tileset_repository

Wikipedia also has a good reference for this character mapping:
https://en.wikipedia.org/wiki/Code_page_437

============  =========  =========  ==================================================
  Tile Index  Unicode    String     Name
============  =========  =========  ==================================================
           0  0x00       \'\\x00\'
           1  0x263A     \'☺\'      WHITE SMILING FACE
           2  0x263B     \'☻\'      BLACK SMILING FACE
           3  0x2665     \'♥\'      BLACK HEART SUIT
           4  0x2666     \'♦\'      BLACK DIAMOND SUIT
           5  0x2663     \'♣\'      BLACK CLUB SUIT
           6  0x2660     \'♠\'      BLACK SPADE SUIT
           7  0x2022     \'•\'      BULLET
           8  0x25D8     \'◘\'      INVERSE BULLET
           9  0x25CB     \'○\'      WHITE CIRCLE
          10  0x25D9     \'◙\'      INVERSE WHITE CIRCLE
          11  0x2642     \'♂\'      MALE SIGN
          12  0x2640     \'♀\'      FEMALE SIGN
          13  0x266A     \'♪\'      EIGHTH NOTE
          14  0x266B     \'♫\'      BEAMED EIGHTH NOTES
          15  0x263C     \'☼\'      WHITE SUN WITH RAYS
          16  0x25BA     \'►\'      BLACK RIGHT-POINTING POINTER
          17  0x25C4     \'◄\'      BLACK LEFT-POINTING POINTER
          18  0x2195     \'↕\'      UP DOWN ARROW
          19  0x203C     \'‼\'      DOUBLE EXCLAMATION MARK
          20  0xB6       \'¶\'      PILCROW SIGN
          21  0xA7       \'§\'      SECTION SIGN
          22  0x25AC     \'▬\'      BLACK RECTANGLE
          23  0x21A8     \'↨\'      UP DOWN ARROW WITH BASE
          24  0x2191     \'↑\'      UPWARDS ARROW
          25  0x2193     \'↓\'      DOWNWARDS ARROW
          26  0x2192     \'→\'      RIGHTWARDS ARROW
          27  0x2190     \'←\'      LEFTWARDS ARROW
          28  0x221F     \'∟\'      RIGHT ANGLE
          29  0x2194     \'↔\'      LEFT RIGHT ARROW
          30  0x25B2     \'▲\'      BLACK UP-POINTING TRIANGLE
          31  0x25BC     \'▼\'      BLACK DOWN-POINTING TRIANGLE
          32  0x20       \' \'      SPACE
          33  0x21       \'!\'      EXCLAMATION MARK
          34  0x22       \'\"\'     QUOTATION MARK
          35  0x23       \'#\'      NUMBER SIGN
          36  0x24       \'$\'      DOLLAR SIGN
          37  0x25       \'%\'      PERCENT SIGN
          38  0x26       \'&\'      AMPERSAND
          39  0x27       \"\'\"     APOSTROPHE
          40  0x28       \'(\'      LEFT PARENTHESIS
          41  0x29       \')\'      RIGHT PARENTHESIS
          42  0x2A       \'\*\'     ASTERISK
          43  0x2B       \'+\'      PLUS SIGN
          44  0x2C       \',\'      COMMA
          45  0x2D       \'-\'      HYPHEN-MINUS
          46  0x2E       \'.\'      FULL STOP
          47  0x2F       \'/\'      SOLIDUS
          48  0x30       \'0\'      DIGIT ZERO
          49  0x31       \'1\'      DIGIT ONE
          50  0x32       \'2\'      DIGIT TWO
          51  0x33       \'3\'      DIGIT THREE
          52  0x34       \'4\'      DIGIT FOUR
          53  0x35       \'5\'      DIGIT FIVE
          54  0x36       \'6\'      DIGIT SIX
          55  0x37       \'7\'      DIGIT SEVEN
          56  0x38       \'8\'      DIGIT EIGHT
          57  0x39       \'9\'      DIGIT NINE
          58  0x3A       \':\'      COLON
          59  0x3B       \';\'      SEMICOLON
          60  0x3C       \'<\'      LESS-THAN SIGN
          61  0x3D       \'=\'      EQUALS SIGN
          62  0x3E       \'>\'      GREATER-THAN SIGN
          63  0x3F       \'?\'      QUESTION MARK
          64  0x40       \'@\'      COMMERCIAL AT
          65  0x41       \'A\'      LATIN CAPITAL LETTER A
          66  0x42       \'B\'      LATIN CAPITAL LETTER B
          67  0x43       \'C\'      LATIN CAPITAL LETTER C
          68  0x44       \'D\'      LATIN CAPITAL LETTER D
          69  0x45       \'E\'      LATIN CAPITAL LETTER E
          70  0x46       \'F\'      LATIN CAPITAL LETTER F
          71  0x47       \'G\'      LATIN CAPITAL LETTER G
          72  0x48       \'H\'      LATIN CAPITAL LETTER H
          73  0x49       \'I\'      LATIN CAPITAL LETTER I
          74  0x4A       \'J\'      LATIN CAPITAL LETTER J
          75  0x4B       \'K\'      LATIN CAPITAL LETTER K
          76  0x4C       \'L\'      LATIN CAPITAL LETTER L
          77  0x4D       \'M\'      LATIN CAPITAL LETTER M
          78  0x4E       \'N\'      LATIN CAPITAL LETTER N
          79  0x4F       \'O\'      LATIN CAPITAL LETTER O
          80  0x50       \'P\'      LATIN CAPITAL LETTER P
          81  0x51       \'Q\'      LATIN CAPITAL LETTER Q
          82  0x52       \'R\'      LATIN CAPITAL LETTER R
          83  0x53       \'S\'      LATIN CAPITAL LETTER S
          84  0x54       \'T\'      LATIN CAPITAL LETTER T
          85  0x55       \'U\'      LATIN CAPITAL LETTER U
          86  0x56       \'V\'      LATIN CAPITAL LETTER V
          87  0x57       \'W\'      LATIN CAPITAL LETTER W
          88  0x58       \'X\'      LATIN CAPITAL LETTER X
          89  0x59       \'Y\'      LATIN CAPITAL LETTER Y
          90  0x5A       \'Z\'      LATIN CAPITAL LETTER Z
          91  0x5B       \'[\'      LEFT SQUARE BRACKET
          92  0x5C       \'\\\\\'   REVERSE SOLIDUS
          93  0x5D       \']\'      RIGHT SQUARE BRACKET
          94  0x5E       \'^\'      CIRCUMFLEX ACCENT
          95  0x5F       \'_\'      LOW LINE
          96  0x60       \'\`\'     GRAVE ACCENT
          97  0x61       \'a\'      LATIN SMALL LETTER A
          98  0x62       \'b\'      LATIN SMALL LETTER B
          99  0x63       \'c\'      LATIN SMALL LETTER C
         100  0x64       \'d\'      LATIN SMALL LETTER D
         101  0x65       \'e\'      LATIN SMALL LETTER E
         102  0x66       \'f\'      LATIN SMALL LETTER F
         103  0x67       \'g\'      LATIN SMALL LETTER G
         104  0x68       \'h\'      LATIN SMALL LETTER H
         105  0x69       \'i\'      LATIN SMALL LETTER I
         106  0x6A       \'j\'      LATIN SMALL LETTER J
         107  0x6B       \'k\'      LATIN SMALL LETTER K
         108  0x6C       \'l\'      LATIN SMALL LETTER L
         109  0x6D       \'m\'      LATIN SMALL LETTER M
         110  0x6E       \'n\'      LATIN SMALL LETTER N
         111  0x6F       \'o\'      LATIN SMALL LETTER O
         112  0x70       \'p\'      LATIN SMALL LETTER P
         113  0x71       \'q\'      LATIN SMALL LETTER Q
         114  0x72       \'r\'      LATIN SMALL LETTER R
         115  0x73       \'s\'      LATIN SMALL LETTER S
         116  0x74       \'t\'      LATIN SMALL LETTER T
         117  0x75       \'u\'      LATIN SMALL LETTER U
         118  0x76       \'v\'      LATIN SMALL LETTER V
         119  0x77       \'w\'      LATIN SMALL LETTER W
         120  0x78       \'x\'      LATIN SMALL LETTER X
         121  0x79       \'y\'      LATIN SMALL LETTER Y
         122  0x7A       \'z\'      LATIN SMALL LETTER Z
         123  0x7B       \'{\'      LEFT CURLY BRACKET
         124  0x7C       \'\|\'     VERTICAL LINE
         125  0x7D       \'}\'      RIGHT CURLY BRACKET
         126  0x7E       \'~\'      TILDE
         127  0x2302     \'⌂\'      HOUSE
         128  0xC7       \'Ç\'      LATIN CAPITAL LETTER C WITH CEDILLA
         129  0xFC       \'ü\'      LATIN SMALL LETTER U WITH DIAERESIS
         130  0xE9       \'é\'      LATIN SMALL LETTER E WITH ACUTE
         131  0xE2       \'â\'      LATIN SMALL LETTER A WITH CIRCUMFLEX
         132  0xE4       \'ä\'      LATIN SMALL LETTER A WITH DIAERESIS
         133  0xE0       \'à\'      LATIN SMALL LETTER A WITH GRAVE
         134  0xE5       \'å\'      LATIN SMALL LETTER A WITH RING ABOVE
         135  0xE7       \'ç\'      LATIN SMALL LETTER C WITH CEDILLA
         136  0xEA       \'ê\'      LATIN SMALL LETTER E WITH CIRCUMFLEX
         137  0xEB       \'ë\'      LATIN SMALL LETTER E WITH DIAERESIS
         138  0xE8       \'è\'      LATIN SMALL LETTER E WITH GRAVE
         139  0xEF       \'ï\'      LATIN SMALL LETTER I WITH DIAERESIS
         140  0xEE       \'î\'      LATIN SMALL LETTER I WITH CIRCUMFLEX
         141  0xEC       \'ì\'      LATIN SMALL LETTER I WITH GRAVE
         142  0xC4       \'Ä\'      LATIN CAPITAL LETTER A WITH DIAERESIS
         143  0xC5       \'Å\'      LATIN CAPITAL LETTER A WITH RING ABOVE
         144  0xC9       \'É\'      LATIN CAPITAL LETTER E WITH ACUTE
         145  0xE6       \'æ\'      LATIN SMALL LETTER AE
         146  0xC6       \'Æ\'      LATIN CAPITAL LETTER AE
         147  0xF4       \'ô\'      LATIN SMALL LETTER O WITH CIRCUMFLEX
         148  0xF6       \'ö\'      LATIN SMALL LETTER O WITH DIAERESIS
         149  0xF2       \'ò\'      LATIN SMALL LETTER O WITH GRAVE
         150  0xFB       \'û\'      LATIN SMALL LETTER U WITH CIRCUMFLEX
         151  0xF9       \'ù\'      LATIN SMALL LETTER U WITH GRAVE
         152  0xFF       \'ÿ\'      LATIN SMALL LETTER Y WITH DIAERESIS
         153  0xD6       \'Ö\'      LATIN CAPITAL LETTER O WITH DIAERESIS
         154  0xDC       \'Ü\'      LATIN CAPITAL LETTER U WITH DIAERESIS
         155  0xA2       \'¢\'      CENT SIGN
         156  0xA3       \'£\'      POUND SIGN
         157  0xA5       \'¥\'      YEN SIGN
         158  0x20A7     \'₧\'      PESETA SIGN
         159  0x0192     \'ƒ\'      LATIN SMALL LETTER F WITH HOOK
         160  0xE1       \'á\'      LATIN SMALL LETTER A WITH ACUTE
         161  0xED       \'í\'      LATIN SMALL LETTER I WITH ACUTE
         162  0xF3       \'ó\'      LATIN SMALL LETTER O WITH ACUTE
         163  0xFA       \'ú\'      LATIN SMALL LETTER U WITH ACUTE
         164  0xF1       \'ñ\'      LATIN SMALL LETTER N WITH TILDE
         165  0xD1       \'Ñ\'      LATIN CAPITAL LETTER N WITH TILDE
         166  0xAA       \'ª\'      FEMININE ORDINAL INDICATOR
         167  0xBA       \'º\'      MASCULINE ORDINAL INDICATOR
         168  0xBF       \'¿\'      INVERTED QUESTION MARK
         169  0x2310     \'⌐\'      REVERSED NOT SIGN
         170  0xAC       \'¬\'      NOT SIGN
         171  0xBD       \'½\'      VULGAR FRACTION ONE HALF
         172  0xBC       \'¼\'      VULGAR FRACTION ONE QUARTER
         173  0xA1       \'¡\'      INVERTED EXCLAMATION MARK
         174  0xAB       \'«\'      LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
         175  0xBB       \'»\'      RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
         176  0x2591     \'░\'      LIGHT SHADE
         177  0x2592     \'▒\'      MEDIUM SHADE
         178  0x2593     \'▓\'      DARK SHADE
         179  0x2502     \'│\'      BOX DRAWINGS LIGHT VERTICAL
         180  0x2524     \'┤\'      BOX DRAWINGS LIGHT VERTICAL AND LEFT
         181  0x2561     \'╡\'      BOX DRAWINGS VERTICAL SINGLE AND LEFT DOUBLE
         182  0x2562     \'╢\'      BOX DRAWINGS VERTICAL DOUBLE AND LEFT SINGLE
         183  0x2556     \'╖\'      BOX DRAWINGS DOWN DOUBLE AND LEFT SINGLE
         184  0x2555     \'╕\'      BOX DRAWINGS DOWN SINGLE AND LEFT DOUBLE
         185  0x2563     \'╣\'      BOX DRAWINGS DOUBLE VERTICAL AND LEFT
         186  0x2551     \'║\'      BOX DRAWINGS DOUBLE VERTICAL
         187  0x2557     \'╗\'      BOX DRAWINGS DOUBLE DOWN AND LEFT
         188  0x255D     \'╝\'      BOX DRAWINGS DOUBLE UP AND LEFT
         189  0x255C     \'╜\'      BOX DRAWINGS UP DOUBLE AND LEFT SINGLE
         190  0x255B     \'╛\'      BOX DRAWINGS UP SINGLE AND LEFT DOUBLE
         191  0x2510     \'┐\'      BOX DRAWINGS LIGHT DOWN AND LEFT
         192  0x2514     \'└\'      BOX DRAWINGS LIGHT UP AND RIGHT
         193  0x2534     \'┴\'      BOX DRAWINGS LIGHT UP AND HORIZONTAL
         194  0x252C     \'┬\'      BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
         195  0x251C     \'├\'      BOX DRAWINGS LIGHT VERTICAL AND RIGHT
         196  0x2500     \'─\'      BOX DRAWINGS LIGHT HORIZONTAL
         197  0x253C     \'┼\'      BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
         198  0x255E     \'╞\'      BOX DRAWINGS VERTICAL SINGLE AND RIGHT DOUBLE
         199  0x255F     \'╟\'      BOX DRAWINGS VERTICAL DOUBLE AND RIGHT SINGLE
         200  0x255A     \'╚\'      BOX DRAWINGS DOUBLE UP AND RIGHT
         201  0x2554     \'╔\'      BOX DRAWINGS DOUBLE DOWN AND RIGHT
         202  0x2569     \'╩\'      BOX DRAWINGS DOUBLE UP AND HORIZONTAL
         203  0x2566     \'╦\'      BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL
         204  0x2560     \'╠\'      BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
         205  0x2550     \'═\'      BOX DRAWINGS DOUBLE HORIZONTAL
         206  0x256C     \'╬\'      BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL
         207  0x2567     \'╧\'      BOX DRAWINGS UP SINGLE AND HORIZONTAL DOUBLE
         208  0x2568     \'╨\'      BOX DRAWINGS UP DOUBLE AND HORIZONTAL SINGLE
         209  0x2564     \'╤\'      BOX DRAWINGS DOWN SINGLE AND HORIZONTAL DOUBLE
         210  0x2565     \'╥\'      BOX DRAWINGS DOWN DOUBLE AND HORIZONTAL SINGLE
         211  0x2559     \'╙\'      BOX DRAWINGS UP DOUBLE AND RIGHT SINGLE
         212  0x2558     \'╘\'      BOX DRAWINGS UP SINGLE AND RIGHT DOUBLE
         213  0x2552     \'╒\'      BOX DRAWINGS DOWN SINGLE AND RIGHT DOUBLE
         214  0x2553     \'╓\'      BOX DRAWINGS DOWN DOUBLE AND RIGHT SINGLE
         215  0x256B     \'╫\'      BOX DRAWINGS VERTICAL DOUBLE AND HORIZONTAL SINGLE
         216  0x256A     \'╪\'      BOX DRAWINGS VERTICAL SINGLE AND HORIZONTAL DOUBLE
         217  0x2518     \'┘\'      BOX DRAWINGS LIGHT UP AND LEFT
         218  0x250C     \'┌\'      BOX DRAWINGS LIGHT DOWN AND RIGHT
         219  0x2588     \'█\'      FULL BLOCK
         220  0x2584     \'▄\'      LOWER HALF BLOCK
         221  0x258C     \'▌\'      LEFT HALF BLOCK
         222  0x2590     \'▐\'      RIGHT HALF BLOCK
         223  0x2580     \'▀\'      UPPER HALF BLOCK
         224  0x03B1     \'α\'      GREEK SMALL LETTER ALPHA
         225  0xDF       \'ß\'      LATIN SMALL LETTER SHARP S
         226  0x0393     \'Γ\'      GREEK CAPITAL LETTER GAMMA
         227  0x03C0     \'π\'      GREEK SMALL LETTER PI
         228  0x03A3     \'Σ\'      GREEK CAPITAL LETTER SIGMA
         229  0x03C3     \'σ\'      GREEK SMALL LETTER SIGMA
         230  0xB5       \'µ\'      MICRO SIGN
         231  0x03C4     \'τ\'      GREEK SMALL LETTER TAU
         232  0x03A6     \'Φ\'      GREEK CAPITAL LETTER PHI
         233  0x0398     \'Θ\'      GREEK CAPITAL LETTER THETA
         234  0x03A9     \'Ω\'      GREEK CAPITAL LETTER OMEGA
         235  0x03B4     \'δ\'      GREEK SMALL LETTER DELTA
         236  0x221E     \'∞\'      INFINITY
         237  0x03C6     \'φ\'      GREEK SMALL LETTER PHI
         238  0x03B5     \'ε\'      GREEK SMALL LETTER EPSILON
         239  0x2229     \'∩\'      INTERSECTION
         240  0x2261     \'≡\'      IDENTICAL TO
         241  0xB1       \'±\'      PLUS-MINUS SIGN
         242  0x2265     \'≥\'      GREATER-THAN OR EQUAL TO
         243  0x2264     \'≤\'      LESS-THAN OR EQUAL TO
         244  0x2320     \'⌠\'      TOP HALF INTEGRAL
         245  0x2321     \'⌡\'      BOTTOM HALF INTEGRAL
         246  0xF7       \'÷\'      DIVISION SIGN
         247  0x2248     \'≈\'      ALMOST EQUAL TO
         248  0xB0       \'°\'      DEGREE SIGN
         249  0x2219     \'∙\'      BULLET OPERATOR
         250  0xB7       \'·\'      MIDDLE DOT
         251  0x221A     \'√\'      SQUARE ROOT
         252  0x207F     \'ⁿ\'      SUPERSCRIPT LATIN SMALL LETTER N
         253  0xB2       \'²\'      SUPERSCRIPT TWO
         254  0x25A0     \'■\'      BLACK SQUARE
         255  0xA0       \'\\xa0\'  NO-BREAK SPACE
============  =========  =========  ==================================================

.. _deprecated-tcod-layout:

Deprecated TCOD Layout
----------------------

The layout for tilesets loaded with: :any:`tcod.tileset.CHARMAP_TCOD`

============  =========  =========  ===========================================
  Tile Index  Unicode    String     Name
============  =========  =========  ===========================================
           0  0x20       \' \'      SPACE
           1  0x21       \'!\'      EXCLAMATION MARK
           2  0x22       \'\"\'     QUOTATION MARK
           3  0x23       \'#\'      NUMBER SIGN
           4  0x24       \'$\'      DOLLAR SIGN
           5  0x25       \'%\'      PERCENT SIGN
           6  0x26       \'&\'      AMPERSAND
           7  0x27       \"\'\"     APOSTROPHE
           8  0x28       \'(\'      LEFT PARENTHESIS
           9  0x29       \')\'      RIGHT PARENTHESIS
          10  0x2A       \'\*\'     ASTERISK
          11  0x2B       \'+\'      PLUS SIGN
          12  0x2C       \',\'      COMMA
          13  0x2D       \'-\'      HYPHEN-MINUS
          14  0x2E       \'.\'      FULL STOP
          15  0x2F       \'/\'      SOLIDUS
          16  0x30       \'0\'      DIGIT ZERO
          17  0x31       \'1\'      DIGIT ONE
          18  0x32       \'2\'      DIGIT TWO
          19  0x33       \'3\'      DIGIT THREE
          20  0x34       \'4\'      DIGIT FOUR
          21  0x35       \'5\'      DIGIT FIVE
          22  0x36       \'6\'      DIGIT SIX
          23  0x37       \'7\'      DIGIT SEVEN
          24  0x38       \'8\'      DIGIT EIGHT
          25  0x39       \'9\'      DIGIT NINE
          26  0x3A       \':\'      COLON
          27  0x3B       \';\'      SEMICOLON
          28  0x3C       \'<\'      LESS-THAN SIGN
          29  0x3D       \'=\'      EQUALS SIGN
          30  0x3E       \'>\'      GREATER-THAN SIGN
          31  0x3F       \'?\'      QUESTION MARK
          32  0x40       \'@\'      COMMERCIAL AT
          33  0x5B       \'[\'      LEFT SQUARE BRACKET
          34  0x5C       \'\\\\\'   REVERSE SOLIDUS
          35  0x5D       \']\'      RIGHT SQUARE BRACKET
          36  0x5E       \'^\'      CIRCUMFLEX ACCENT
          37  0x5F       \'_\'      LOW LINE
          38  0x60       \'\`\'     GRAVE ACCENT
          39  0x7B       \'{\'      LEFT CURLY BRACKET
          40  0x7C       \'\|\'     VERTICAL LINE
          41  0x7D       \'}\'      RIGHT CURLY BRACKET
          42  0x7E       \'~\'      TILDE
          43  0x2591     \'░\'      LIGHT SHADE
          44  0x2592     \'▒\'      MEDIUM SHADE
          45  0x2593     \'▓\'      DARK SHADE
          46  0x2502     \'│\'      BOX DRAWINGS LIGHT VERTICAL
          47  0x2500     \'─\'      BOX DRAWINGS LIGHT HORIZONTAL
          48  0x253C     \'┼\'      BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
          49  0x2524     \'┤\'      BOX DRAWINGS LIGHT VERTICAL AND LEFT
          50  0x2534     \'┴\'      BOX DRAWINGS LIGHT UP AND HORIZONTAL
          51  0x251C     \'├\'      BOX DRAWINGS LIGHT VERTICAL AND RIGHT
          52  0x252C     \'┬\'      BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
          53  0x2514     \'└\'      BOX DRAWINGS LIGHT UP AND RIGHT
          54  0x250C     \'┌\'      BOX DRAWINGS LIGHT DOWN AND RIGHT
          55  0x2510     \'┐\'      BOX DRAWINGS LIGHT DOWN AND LEFT
          56  0x2518     \'┘\'      BOX DRAWINGS LIGHT UP AND LEFT
          57  0x2598     \'▘\'      QUADRANT UPPER LEFT
          58  0x259D     \'▝\'      QUADRANT UPPER RIGHT
          59  0x2580     \'▀\'      UPPER HALF BLOCK
          60  0x2596     \'▖\'      QUADRANT LOWER LEFT
          61  0x259A     \'▚\'      QUADRANT UPPER LEFT AND LOWER RIGHT
          62  0x2590     \'▐\'      RIGHT HALF BLOCK
          63  0x2597     \'▗\'      QUADRANT LOWER RIGHT
          64  0x2191     \'↑\'      UPWARDS ARROW
          65  0x2193     \'↓\'      DOWNWARDS ARROW
          66  0x2190     \'←\'      LEFTWARDS ARROW
          67  0x2192     \'→\'      RIGHTWARDS ARROW
          68  0x25B2     \'▲\'      BLACK UP-POINTING TRIANGLE
          69  0x25BC     \'▼\'      BLACK DOWN-POINTING TRIANGLE
          70  0x25C4     \'◄\'      BLACK LEFT-POINTING POINTER
          71  0x25BA     \'►\'      BLACK RIGHT-POINTING POINTER
          72  0x2195     \'↕\'      UP DOWN ARROW
          73  0x2194     \'↔\'      LEFT RIGHT ARROW
          74  0x2610     \'☐\'      BALLOT BOX
          75  0x2611     \'☑\'      BALLOT BOX WITH CHECK
          76  0x25CB     \'○\'      WHITE CIRCLE
          77  0x25C9     \'◉\'      FISHEYE
          78  0x2551     \'║\'      BOX DRAWINGS DOUBLE VERTICAL
          79  0x2550     \'═\'      BOX DRAWINGS DOUBLE HORIZONTAL
          80  0x256C     \'╬\'      BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL
          81  0x2563     \'╣\'      BOX DRAWINGS DOUBLE VERTICAL AND LEFT
          82  0x2569     \'╩\'      BOX DRAWINGS DOUBLE UP AND HORIZONTAL
          83  0x2560     \'╠\'      BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
          84  0x2566     \'╦\'      BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL
          85  0x255A     \'╚\'      BOX DRAWINGS DOUBLE UP AND RIGHT
          86  0x2554     \'╔\'      BOX DRAWINGS DOUBLE DOWN AND RIGHT
          87  0x2557     \'╗\'      BOX DRAWINGS DOUBLE DOWN AND LEFT
          88  0x255D     \'╝\'      BOX DRAWINGS DOUBLE UP AND LEFT
          89  0x00       \'\\x00\'
          90  0x00       \'\\x00\'
          91  0x00       \'\\x00\'
          92  0x00       \'\\x00\'
          93  0x00       \'\\x00\'
          94  0x00       \'\\x00\'
          95  0x00       \'\\x00\'
          96  0x41       \'A\'      LATIN CAPITAL LETTER A
          97  0x42       \'B\'      LATIN CAPITAL LETTER B
          98  0x43       \'C\'      LATIN CAPITAL LETTER C
          99  0x44       \'D\'      LATIN CAPITAL LETTER D
         100  0x45       \'E\'      LATIN CAPITAL LETTER E
         101  0x46       \'F\'      LATIN CAPITAL LETTER F
         102  0x47       \'G\'      LATIN CAPITAL LETTER G
         103  0x48       \'H\'      LATIN CAPITAL LETTER H
         104  0x49       \'I\'      LATIN CAPITAL LETTER I
         105  0x4A       \'J\'      LATIN CAPITAL LETTER J
         106  0x4B       \'K\'      LATIN CAPITAL LETTER K
         107  0x4C       \'L\'      LATIN CAPITAL LETTER L
         108  0x4D       \'M\'      LATIN CAPITAL LETTER M
         109  0x4E       \'N\'      LATIN CAPITAL LETTER N
         110  0x4F       \'O\'      LATIN CAPITAL LETTER O
         111  0x50       \'P\'      LATIN CAPITAL LETTER P
         112  0x51       \'Q\'      LATIN CAPITAL LETTER Q
         113  0x52       \'R\'      LATIN CAPITAL LETTER R
         114  0x53       \'S\'      LATIN CAPITAL LETTER S
         115  0x54       \'T\'      LATIN CAPITAL LETTER T
         116  0x55       \'U\'      LATIN CAPITAL LETTER U
         117  0x56       \'V\'      LATIN CAPITAL LETTER V
         118  0x57       \'W\'      LATIN CAPITAL LETTER W
         119  0x58       \'X\'      LATIN CAPITAL LETTER X
         120  0x59       \'Y\'      LATIN CAPITAL LETTER Y
         121  0x5A       \'Z\'      LATIN CAPITAL LETTER Z
         122  0x00       \'\\x00\'
         123  0x00       \'\\x00\'
         124  0x00       \'\\x00\'
         125  0x00       \'\\x00\'
         126  0x00       \'\\x00\'
         127  0x00       \'\\x00\'
         128  0x61       \'a\'      LATIN SMALL LETTER A
         129  0x62       \'b\'      LATIN SMALL LETTER B
         130  0x63       \'c\'      LATIN SMALL LETTER C
         131  0x64       \'d\'      LATIN SMALL LETTER D
         132  0x65       \'e\'      LATIN SMALL LETTER E
         133  0x66       \'f\'      LATIN SMALL LETTER F
         134  0x67       \'g\'      LATIN SMALL LETTER G
         135  0x68       \'h\'      LATIN SMALL LETTER H
         136  0x69       \'i\'      LATIN SMALL LETTER I
         137  0x6A       \'j\'      LATIN SMALL LETTER J
         138  0x6B       \'k\'      LATIN SMALL LETTER K
         139  0x6C       \'l\'      LATIN SMALL LETTER L
         140  0x6D       \'m\'      LATIN SMALL LETTER M
         141  0x6E       \'n\'      LATIN SMALL LETTER N
         142  0x6F       \'o\'      LATIN SMALL LETTER O
         143  0x70       \'p\'      LATIN SMALL LETTER P
         144  0x71       \'q\'      LATIN SMALL LETTER Q
         145  0x72       \'r\'      LATIN SMALL LETTER R
         146  0x73       \'s\'      LATIN SMALL LETTER S
         147  0x74       \'t\'      LATIN SMALL LETTER T
         148  0x75       \'u\'      LATIN SMALL LETTER U
         149  0x76       \'v\'      LATIN SMALL LETTER V
         150  0x77       \'w\'      LATIN SMALL LETTER W
         151  0x78       \'x\'      LATIN SMALL LETTER X
         152  0x79       \'y\'      LATIN SMALL LETTER Y
         153  0x7A       \'z\'      LATIN SMALL LETTER Z
         154  0x00       \'\\x00\'
         155  0x00       \'\\x00\'
         156  0x00       \'\\x00\'
         157  0x00       \'\\x00\'
         158  0x00       \'\\x00\'
         159  0x00       \'\\x00\'
============  =========  =========  ===========================================
