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
##    Pete Shinners
##    pete@shinners.org

"sysfont, used in the font module to find system fonts"

import os, sys


#create simple version of the font name
def _simplename(name):
    for char in '_ -':
        name = name.replace(char, '')
    name = name.lower()
    name = name.replace('-', '')
    name = name.replace("'", '')
    return name


#insert a font and style into the font dictionary
def _addfont(name, bold, italic, font, fontdict):
    if not fontdict.has_key(name):
        fontdict[name] = {}
    fontdict[name][bold, italic] = font


#read the fonts on windows
def initsysfonts_win32():
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

            bold = italic = 0
            for m in mods:
                if m in name:
                    name.remove(m)
            if 'bold' in name:
                name.remove('bold')
                bold = 1
            if 'italic' in name:
                name.remove('italic')
                italic = 1
            name = ''.join(name)

            name=_simplename(name)

            _addfont(name, bold, italic, font, fonts)
    return fonts


#read of the fonts on osx (fill me in!)
def initsysfonts_darwin():
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




#read the fonts on unix
def initsysfonts_unix():
    fonts = {}

    # we use the fc-list from fontconfig to get a list of fonts.

    try:
        # note, we use popen3 for if fc-list isn't there to stop stderr printing.
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
                    _addfont(_simplename(family), bold, italic or oblique, filename, fonts)
            except:
                # try the next one.
                pass
    except:
        pass

    return fonts



#create alias entries
def create_aliases():
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


Sysfonts = {}
Sysalias = {}

#initialize it all, called once
def initsysfonts():
    if sys.platform == 'win32':
        fonts = initsysfonts_win32()
    elif sys.platform == 'darwin':
        fonts = initsysfonts_darwin()
    else:
        fonts = initsysfonts_unix()
    Sysfonts.update(fonts)
    create_aliases()
    if not Sysfonts: #dummy so we don't try to reinit
        Sysfonts[None] = None



#the exported functions

def SysFont(name, size, bold=False, italic=False):
    '''Create a Font object from the system fonts.

    Return a new Font object that is loaded from the system fonts. The font will
    match the requested bold and italic flags. If a suitable system font is not
    found this will fallback on loading the default pygame font. The font name
    can be a comma separated list of font names to look for.
    
    :Parameters:
        `name` : str
            Font family or comma-separated list of families.
        `size` : int
            Size of font, in points.
        `bold` : bool
            True if boldface variant requested.
        `italic` : bool
            True if italic variant requested.

    :rtype: `Font`
    '''
    import pygame.font

    if not Sysfonts:
        initsysfonts()
    
    gotbold = gotitalic = False
    fontname = None
    if name:
        allnames = name
        for name in allnames.split(','):
            name = _simplename(name)
            styles = Sysfonts.get(name)
            if not styles:
                styles = Sysalias.get(name)
            if styles:
                while not fontname:
                    plainname = styles.get((False, False))
                    fontname = styles.get((bold, italic))
                    if plainname != fontname:
                        gotbold = bold
                        gotitalic = italic
                    elif not fontname:
                        fontname = plainname
            if fontname: break

    font = pygame.font.Font(fontname, size)
    if bold and not gotbold:
        font.set_bold(1)
    if italic and not gotitalic:
        font.set_italic(1)

    return font


def get_fonts():
    '''Get all available fonts.

    Returns a list of all the fonts available on the system. The names of the
    fonts will be set to lowercase with all spaces and punctuation removed.
    This works on most systems, but some will return an empty list if they
    cannot find fonts.  

    :rtype: list of str
    '''

    if not Sysfonts:
        initsysfonts()
    return Sysfonts.keys()


def match_font(name, bold=0, italic=0):
    '''Find a specific font on the system.

    Returns the full path to a font file on the system. If bold or italic are
    set to true, this will attempt to find the correct family of font.

    The font name can actually be a comma separated list of font names to try.
    If none of the given names are found, None is returned.

    Example::

        >>> print pygame.font.match_font('bitstreamverasans')
        '/usr/share/fonts/truetype/ttf-bitstream-vera/Vera.ttf'
    
    :Parameters:
        `name` : str
            Font family or comma-separated list of families.
        `bold` : bool
            True if boldface variant requested.
        `italic` : bool
            True if italic variant requested.

    :rtype: str
    '''
    if not Sysfonts:
        initsysfonts()

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
                    italic = 0
                elif bold:
                    bold = 0
                elif not fontname:
                    fontname = styles.values()[0]
        if fontname: break
    return fontname


