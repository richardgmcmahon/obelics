def write_ds9_regionfile(ralist, declist, filename='ds9.reg', symbol='circle',
                         color='green',
                         comments=None,
                         text=None):


    """Write out a DS9 region file.
    ;
    ; Python version of write_ds9_regionfile.pro
    ;
    ; INPUTS:
    ;   ra  - right ascension
    ;   dec - declination
    ;
    ; OPTIONAL INPUTS:
    ;   filename - output file name (default 'ds9.reg')
    ;   symbol   - symbol to draw; supported symbols are: 'box',
    ;              'diamond', 'cross', 'x', 'arrow', and 'boxcircle'
    ;              (default 'circle')
    ;   color    - region color; supported colors are 'white',
    ;              'black', 'red', 'green', 'blue', 'cyan', 'magenta',
    ;              and 'yellow' (default 'green')
    ;
    ;   text     - string e.g. integer counter for each region
    ;
    ; KEYWORD PARAMETERS:
    ;
    ; OUTPUTS:
    ;
    ; OPTIONAL OUTPUTS:
    ;
    ; COMMENTS:
    ;
    ; EXAMPLES:
    ;
    ; MODIFICATION HISTORY:
    ;   J. Moustakas, 2007 Jun 18, NYU - written
    ;   jm08jun09nyu - added SYMBOL optional input
    ;
    ;    2012-12-15 - rgm: added support for text
    ;
    ; Copyright (C) 2007-2008, John Moustakas
    ; Copyright (C) 2012,  Richard McMahon
    ;
    ; This program is free software; you can redistribute it and/or modify
    ; it under the terms of the GNU General Public License as published by
    ; the Free Software Foundation; either version 2 of the License, or
    ; (at your option) any later version.
    ;
    ; This program is distributed in the hope that it will be useful, but
    ; WITHOUT ANY WARRANTY; without even the implied warranty of
    ; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    ; General Public License for more details.
    ;-

    """

    nobjects = len(ralist)

    fh = open(filename, "w")

    if comments is not None:
        fh.write(comments + '\n')

    fh.write('global color=' + color + ' font="helvetica 10 normal" ' + \
      'select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source\n')
    fh.write('fk5\n')


    for i, ra in enumerate(ralist):

        string = 'point(' + str(ralist[i]) + ',' \
                 + str(declist[i]) + ')' \
                 + ' # point=' + symbol

        if text is not None:
            string = string + ' text={' + text[i]  +'}'

        fh.write(string + '\n')

    fh.close()

    return
