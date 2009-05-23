##    pygame - Python Game Library
##    Copyright (C) 2000-2003  Pete Shinners
##
##    This library is free software; you can redistribute it and/or
##    modify it under the terms of the GNU Library General Public
##    License as published by the Free Software Foundation; either
##    version 2 of the License, or (at your option) any later version.
##
##    This library is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
##    Library General Public License for more details.
##
##    You should have received a copy of the GNU Library General Public
##    License along with this library; if not, write to the Free
##    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##

"font, used to find system fonts"

import os, sys

Sysfonts = {}
Sysalias = {}

def _simplename(name):
    """
    Create simple version of the font name
    """
    for char in '_ -':
        name = name.replace(char, '')
    name = name.lower()
    name = name.replace('-', '')
    name = name.replace("'", '')
    return name

def _addfont(name, bold, italic, font, fontdict):
    """
    insert a font and style into the font dictionary
    """
    if not fontdict.has_key(name):
        fontdict[name] = {}
    fontdict[name][bold, italic] = font


def _initsysfonts_win32():
    """
    read the fonts on windows
    """
    import _winreg
    fonts = {}
    mods = 'demibold', 'narrow', 'light', 'unicode', 'bt', 'mt'
    fontdir = os.path.join(os.environ['WINDIR'], "Fonts")

    #this is a list of registry keys containing information
    #about fonts installed on the system.
    keys = []

    #find valid registry keys containing font information.
    possible_keys = [
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\Fonts",
        r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"
        ]

    for key_name in possible_keys:
        try:
            key = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, key_name)
            keys.append(key)
        except WindowsError:
            pass

    for key in keys:
        fontdict = {}
        for i in range(_winreg.QueryInfoKey(key)[1]):
            try: name, font, t = _winreg.EnumValue(key,i)
            except EnvironmentError: break

            # try and handle windows unicode strings for some file names.
            
            # here are two documents with some information about it:
            # http://www.python.org/peps/pep-0277.html
            # https://www.microsoft.com/technet/archive/interopmigration/linux/mvc/lintowin.mspx#ECAA
            try:
                font = str(font)
            except UnicodeEncodeError:
                # MBCS is the windows encoding for unicode file names.
                try:
                    font = font.encode('MBCS')
                except:
                    # no goodness with str or MBCS encoding... skip this font.
                    continue
   
            if font[-4:].lower() not in [".ttf", ".ttc"]:
                continue
            if os.sep not in font:
                font = os.path.join(fontdir, font)

            if name[-10:] == '(TrueType)':
                name = name[:-11]
            name = name.lower().split()

            bold = italic = False
            for m in mods:
                if m in name:
                    name.remove(m)
            if 'bold' in name:
                name.remove('bold')
                bold = True
            if 'italic' in name:
                name.remove('italic')
                italic = True
            name = ''.join(name)

            name=_simplename(name)

            _addfont(name, bold, italic, font, fonts)
    return fonts

def _initsysfonts_darwin():
    """
    read of the fonts on osx (fill me in!)
    """
    paths = ['/Library/Fonts',
             '~/Library/Fonts',
             '/Local/Library/Fonts',
             '/Network/Library/Fonts']
    fonts = {}
    for p in paths:
        if os.path.isdir(p):
            pass
            #os.path.walk(p, _fontwalk, fonts)
    return fonts

def _initsysfonts_unix():
    """
    read the fonts on unix
    """
    fonts = {}

    # we use the fc-list from fontconfig to get a list of fonts.

    try:
        # note, we use popen3 for if fc-list isn't there to stop stderr
        # printing.
        flin, flout, flerr = os.popen3('fc-list : file family style')
    except:
        return fonts

    try:
        for line in flout:
            try:
                filename, family, style = line.split(':', 2)
                if filename[-4:].lower() in ['.ttf', '.ttc']:
                    bold = style.find('Bold') >= 0
                    italic = style.find('Italic') >= 0
                    oblique = style.find('Oblique') >= 0
                    _addfont(_simplename(family), bold, italic or \
                             oblique, filename, fonts)
            except:
                # try the next one.
                pass
    except:
        pass

    return fonts

def _create_aliases():
    """
    create alias entries
    """
    aliases = (
        ('monospace', 'misc-fixed', 'courier', 'couriernew', 'console',
         'fixed', 'mono', 'freemono', 'bitstreamverasansmono',
         'verasansmono', 'monotype', 'lucidaconsole'),
        ('sans', 'arial', 'helvetica', 'swiss', 'freesans',
         'bitstreamverasans', 'verasans', 'verdana', 'tahoma'),
        ('serif', 'times', 'freeserif', 'bitstreamveraserif', 'roman',
         'timesroman', 'timesnewroman', 'dutch', 'veraserif',
         'georgia'),
        ('wingdings', 'wingbats'),
    )
    for set in aliases:
        found = None
        fname = None
        for name in set:
            if Sysfonts.has_key(name):
                found = Sysfonts[name]
                fname = name
                break
        if not found:
            continue
        for name in set:
            if not Sysfonts.has_key(name):
                Sysalias[name] = found

def _initsysfonts():
    """
    initialize it all, called once
    """
    if sys.platform == 'win32':
        fonts = _initsysfonts_win32()
    elif sys.platform == 'darwin':
        fonts = _initsysfonts_darwin()
    else:
        fonts = _initsysfonts_unix()
    Sysfonts.update(fonts)
    _create_aliases()
    if not Sysfonts: #dummy so we don't try to reinit
        Sysfonts[None] = None

#the exported functions

def get_fonts():
    """
    get_fonts () -> list
    
    Get a list of system font names.

    Returns the list of all found system fonts. Note that the names of the
    fonts will be all lowercase with spaces removed. This is how pygame2
    internally stores the font names for matching.
    """
    if not Sysfonts:
        _initsysfonts()
    return Sysfonts.keys()

def find_font(name, bold=False, italic=False):
    """
    find_font (name, bold=False, italic=False) -> str, bool, bool
    
    Find the filename for the named system font.

    This performs a font search and it returns the path to the TTF file.
    The font name can be a comma separated list of font names to try.
    If no match is found, None is returned as fontname.
    """
    if not Sysfonts:
        _initsysfonts()

    fontname = None
    allnames = name
    for name in allnames.split(','):
        name = _simplename(name)
        styles = Sysfonts.get(name)
        if not styles:
            styles = Sysalias.get(name)
        if styles:
            while not fontname:
                fontname = styles.get((bold, italic))
                if italic:
                    italic = False
                elif bold:
                    bold = False
                elif not fontname:
                    fontname = styles.values()[0]
        if fontname:
            break
    return fontname, italic, bold

__all__ = [ "find_font", "get_fonts" ]
