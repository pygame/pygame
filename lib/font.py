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

import glob, os, sys, subprocess

#
# font styles consist of (fullname, filetype, bold, italic)
#

_fonts = {}
_aliases = {}
_families = {}

# Fonts on windows
# Info taken from:
# http://www.microsoft.com/typography/fonts/winxp.htm
# with extra files added from:
# http://www.ampsoft.net/webdesign-l/windows-fonts-by-version.html
# File name, family, (Bold, Italic)
_win32_fontfiles = [
    ('ahronbd.ttf', 'Aharoni', True, False),
    ('andlso.ttf', 'Andalus', False, False),
    ('angsa.ttf', 'Angsana New', False, False),
    ('angsab.ttf', 'Angsana New', True, False),
    ('angsai.ttf', 'Angsana New', False, True),
    ('angsaz.ttf', 'Angsana New', True, True),
    ('angsau.ttf', 'AngsanaUPC', False, False),
    ('angsaub.ttf', 'AngsanaUPC', True, False),
    ('angsaui.ttf', 'AngsanaUPC', False, True),
    ('angsauz.ttf', 'AngsanaUPC', True, True),
    ('artro.ttf', 'Arabic Transparent', False, False),
    ('artrbdo.ttf', 'Arabic Transparent', True, False),
    ('agatha.ttf', 'Agatha', False, False),
    ('arial.ttf', 'Arial', False, False),
    ('arialbd.ttf', 'Arial', True, False),
    ('ariali.ttf', 'Arial', False, True),
    ('arialbi.ttf', 'Arial', True, True),
    ('ariblk.ttf', 'Arial Black', False, False),
    ('browa.ttf', 'Browallia New', False, False),
    ('browab.ttf', 'Browallia New', True, False),
    ('browai.ttf', 'Browallia New', False, True),
    ('browaz.ttf', 'Browallia New', True, True),
    ('browau.ttf', 'BrowalliaUPC', False, False),
    ('browaub.ttf', 'BrowalliaUPC', True, False),
    ('browaui.ttf', 'BrowalliaUPC', False, True),
    ('browauz.ttf', 'BrowalliaUPC', True, True),
    ('comic.ttf', 'Comic Sans MS', False, False),
    ('comicbd.ttf', 'Comic Sans MS', True, False),
    ('cordia.ttf', 'Cordia New', False, False),
    ('cordiab.ttf', 'Cordia New', True, False),
    ('cordiai.ttf', 'Cordia New', False, True),
    ('cordiaz.ttf', 'Cordia New', True, True),
    ('cordiau.ttf', 'CordiaUPC', False, False),
    ('cordiaub.ttf', 'CordiaUPC', True, False),
    ('cordiaui.ttf', 'CordiaUPC', False, True),
    ('cordiauz.ttf', 'CordiaUPC', True, True),
    ('cour.ttf', 'Courier New', False, False),
    ('courbd.ttf', 'Courier New', True, False),
    ('couri.ttf', 'Courier New', False, True),
    ('courbi.ttf', 'Courier New', True, True),
    ('david.ttf', 'David', False, False),
    ('davidbd.ttf', 'David', True, False),
    ('davidtr.ttf', 'David Transparent', False, False),
    ('upcdl.ttf', 'DilleniaUPC', False, False),
    ('upcdb.ttf', 'DilleniaUPC', True, False),
    ('upcdi.ttf', 'DilleniaUPC', False, True),
    ('upcdbi.ttf', 'DilleniaUPC', True, True),
    ('estre.ttf', 'Estrangelo Edessa', False, False),
    ('upcel.ttf', 'EucrosialUPC', False, False),
    ('upceb.ttf', 'EucrosialUPC', True, False),
    ('upcei.ttf', 'EucrosialUPC', False, True),
    ('upcebi.ttf', 'EucrosialUPC', True, True),
    ('mriamfx.ttf', 'Fixed Miriam Transparent', False, False),
    ('framd.ttf', 'Franklin Gothic Medium', False, False),
    ('framdit.ttf', 'Franklin Gothic Medium', False, True),
    ('frank.ttf', 'FrankRuehl', False, False),
    ('upcfl.ttf', 'FreesialUPC', False, False),
    ('upcfb.ttf', 'FreesialUPC', True, False),
    ('upcfi.ttf', 'FreesialUPC', False, True),
    ('upcfbi.ttf', 'FreesialUPC', True, True),
    ('gautami.ttf', 'Gautami', False, False),
    ('georgia.ttf', 'Georgia', False, False),
    ('georgiab.ttf', 'Georgia', True, False),
    ('georgiai.ttf', 'Georgia', False, True),
    ('georgiaz.ttf', 'Georgia', True, True),
    ('impact.ttf', 'Impact', False, False),
    ('upcil.ttf', 'IrisUPC', False, False),
    ('upcib.ttf', 'IrisUPC', True, False),
    ('upcii.ttf', 'IrisUPC', False, True),
    ('upcibi.ttf', 'IrisUPC', True, True),
    ('upcjl.ttf', 'JasmineUPC', False, False),
    ('upcjb.ttf', 'JasmineUPC', True, False),
    ('upcji.ttf', 'JasmineUPC', False, True),
    ('upcjbi.ttf', 'JasmineUPC', True, True),
    ('upckl.ttf', 'KodchiangUPC', False, False),
    ('upckb.ttf', 'KodchiangUPC', True, False),
    ('upcki.ttf', 'KodchiangUPC', False, True),
    ('upckbi.ttf', 'KodchiangUPC', True, True),
    ('latha.ttf', 'Latha', False, False),
    ('lvnm.ttf', 'Levenim MT', False, False),
    ('lvnmbd.ttf', 'Levenim MT', True, False),
    ('upcll.ttf', 'LilyUPC', False, False),
    ('upclb.ttf', 'LilyUPC', True, False),
    ('upcli.ttf', 'LilyUPC', False, True),
    ('upclbi.ttf', 'LilyUPC', True, True),
    ('lucon.ttf', 'Lucida Console', False, False),
    ('l_10646.ttf', 'Lucida Sans Unicode', False, False),
    ('mangal.ttf', 'Mangal', False, False),
    ('marlett.ttf', 'Marlett', False, False),
    ('micross.ttf', 'Microsoft Sans Serif', False, False),
    ('mriam.ttf', 'Miriam', False, False),
    ('mriamc.ttf', 'Miriam Fixed', False, False),
    ('mriamtr.ttf', 'Miriam Transparent', False, False),
    ('mvboli.ttf', 'MV Boli', False, False),
    ('nrkis.ttf', 'Narkisim', False, False),
    ('pala.ttf', 'Falatino Linotype', False, False),
    ('palab.ttf', 'Falatino Linotype', True, False),
    ('palai.ttf', 'Falatino Linotype', False, True),
    ('palabi.ttf', 'Falatino Linotype', True, True),
    ('raavi.ttf', 'Raavi', False, False),
    ('rod.ttf', 'Rod', False, False),
    ('rodtr.ttf', 'Rod Transparent', False, False),
    ('shruti.ttf', 'Shruti', False, False),
    ('simpo.ttf', 'Simplified Arabic', False, False),
    ('simpbdo.ttf', 'Simplified Arabic', True, False),
    ('simpfxo.ttf', 'Simplified Arabic Fixed', False, False),
    ('sylfaen.ttf', 'Sylfaen', False, False),
    ('symbol.ttf', 'Symbol', False, False),
    ('tahoma.ttf', 'Tahoma', False, False),
    ('tahomabd.ttf', 'Tahoma', True, False),
    ('times.ttf', 'Times New Roman', False, False),
    ('timesbd.ttf', 'Times New Roman', True, False),
    ('timesi.ttf', 'Times New Roman', False, True),
    ('timesbi.ttf', 'Times New Roman', True, True),
    ('trado.ttf', 'Traditional Arabic', False, False),
    ('tradbdo.ttf', 'Traditional Arabic', True, False),
    ('Trebuc.ttf', 'Trebuchet MS', False, False),
    ('Trebucbd.ttf', 'Trebuchet MS', True, False),
    ('Trebucit.ttf', 'Trebuchet MS', False, True),
    ('Trebucbi.ttf', 'Trebuchet MS', True, True),
    ('tunga.ttf', 'Tunga', False, False),
    ('verdana.ttf', 'Verdana', False, False),
    ('verdanab.ttf', 'Verdana', True, False),
    ('verdanai.ttf', 'Verdana', False, True),
    ('verdanaz.ttf', 'Verdana', True, True),
    ('webdings.ttf', 'Webdings', False, False),
    ('wingding.ttf', 'Wingdings', False, False),
    ('simhei.ttf', 'SimHei', False, False),
    ('simfang.ttf', 'FangSong_GB2312', False, False),
    ('kaiu.ttf', 'DFKai-SB', False, False),
    ('simkai.ttf', 'KaiTi_GB2312', False, False),
    ('msgothic.ttc', 'MS Gothic', False, False),
    ('msmincho.ttc', 'MS Mincho', False, False),
    ('gulim.ttc', 'Gulim', False, False),
    ('mingliu.ttc', 'Mingliu', False, False),
    ('simsun.ttc', 'Simsun', False, False),
    ('batang.ttc', 'Batang', False, False),
    ]

def _simplename (name):
    """_simplename (name) -> str

    Simplifies a font name, removing any characters not being alphanumeric.
    """
    return ''.join ([c.lower () for c in name if c.isalnum ()])

def _addfont (filename, name, ftype, family, bold, italic):
    """_addfont (filename, name, ftype, family, bold, italic) -> None

    Adds a font and its family to the internal font file caches.
    """
    family = _simplename (family)
    if family not in _families:
        _families[family] = []
    if name in _fonts:
        print ("*** Font %s will be overwritten" % name)
    _fonts[filename] = (name, ftype, bold, italic)
    _families[family].append (filename)

def _gettype (filename):
    """_gettype (filename) -> str

    Gets the type of the font file, based on its file extension.
    """
    ftype = os.path.splitext (filename)[1].lstrip (".")
    if filename.endswith (".pcf.gz"):
        ftype = "pcf"
    return ftype.lower ()

def _getstyle (style):
    """_getstyle (style) -> bool, bool

    Gets the bold and italic style properties from the passed style.
    """
    bold = italic = oblique = False
    if sys.platform == "win32":
        pass
    else:
        bold = style.find ("Bold") >= 0
        italic = style.find ("Italic") >= 0
        oblique = style.find ("Oblique") >= 0
    return bold, italic or oblique

def _initwin32 ():
    """_initwin32 () -> None
    
    Initializes the win32-based font cache.
    """
    fontdir = "C:\\Windows\\Fonts"
    if "WINDIR" in os.environ:
        fontdir = os.path.join (os.environ["WINDIR"], "Fonts")
    elif "windir" in os.environ:
        fontdir = os.path.join (os.environ["windir"], "Fonts")
    
    # Build the lookup table.
    lookups = dict ([(fname.lower (), (_simplename(name), bold, italic))
                     for fname, name, bold, italic in _win32_fontfiles])
    
    # Walk over the windows font file directory first.
    files = glob.glob (os.path.join (fontdir, "*"))
    for font in files:
        font = font.lower ()
        filename = os.path.basename (font)
        name = os.path.splitext (filename)[0]
        ftype = _gettype (filename)
        family = ""
        bold = italic = False
        if filename in lookups:
            name, bold, italic = lookups[filename]
        _addfont (font, name, ftype, family, bold, italic)

    # Lookup any mappings existing in the registry.
    try:
        import _winreg
    except ImportError:
        try:
            import winreg as _winreg
        except ImportError:
            return

    possible_keys = [
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\Fonts",
        r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts",
        r"SOFTWARE\Microsoft\Windows[NT]\CurrentVersion\Fonts",
        ]
    keys = []
    for key_name in possible_keys:
        try:
            key = _winreg.OpenKey (_winreg.HKEY_LOCAL_MACHINE, key_name)
            keys.append (key)
        except WindowsError:
            pass

    for key in keys:
        fontdict = {}
        for i in range (_winreg.QueryInfoKey (key)[1]):
            name, font, t = None, None, None
            try:
                name, font, t = _winreg.EnumValue (key, i)
            except EnvironmentError:
                break

            # try and handle windows unicode strings for some file names.
            
            # here are two documents with some information about it:
            # http://www.python.org/peps/pep-0277.html
            # https://www.microsoft.com/technet/archive/interopmigration/linux/mvc/lintowin.mspx#ECAA
            try:
                font = str (font)
            except UnicodeEncodeError:
                # MBCS is the windows encoding for unicode file names.
                try:
                    font = font.encode ('MBCS')
                except:
                    # no goodness with str or MBCS encoding... skip this font.
                    continue
            if os.path.sep not in font:
                font = os.path.join (fontdir, font)
            font = font.lower ()
            if font in _fonts:
                continue # Skip the font, if already detected.
            if name.find ("(") != -1:
                name = name[:name.find ("(")].rstrip ()
            name = name.lower ()

def _initunix ():
    """_initunix () -> None

    Initializes the unix-based font cache using the fontconfig utilities.
    """
    output = ""
    try:
        p = subprocess.Popen \
		("fc-list : file family style fullname fullnamelang",
                 shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = p.communicate()[0]
        retcode = p.returncode
        if sys.version_info[0] >= 3:
            output = str (output, "utf-8")
    except OSError:
        _families[None] = None
        _fonts[None] = None
        return

    try:
        for line in output.split ("\n"):
            if line.strip () == "":
               continue 
            values = line.split (":")
            fullnames, fulllang, name = None, None, None
            filename, family, style = values[0:3]
            if len (values) > 3:
                pos = -1
                fullnames, fulllang = values[3:]
                langs = fulllang.split (",")
                if "fullnamelang=en" in langs:
                    pos = langs.index ("fullnamelang=en")
                else:
                    pos = langs.index ("en")
                if pos != -1:
                    name = fullnames.split (",")[pos].lstrip ("fullname=")
                    
            else:
                name = os.path.splitext (os.path.basename (filename))[0]
                if name.endswith (".pcf"):
                    name = name[:-4]
                name = _simplename (name)
            ftype = _gettype (filename)
            bold, italic = _getstyle (style)
            _addfont (filename, name, ftype, family, bold, italic)
    except Exception:
        _families.clear ()
        _fonts.clear ()
        _families[None] = None
        _fonts[None] = None

def _initfonts ():
    """_initfonts () -> None

    Initializes the internal font caches.
    """
    if _fonts:
        return
    if sys.platform == "win32":
        _initwin32 ()
    elif sys.platform == "darwin":
        # TODO
        pass
    else:
        _initunix ()

def get_families ():
    """get_families () -> [str, str, str, ...]

    Gets the list of available font families.
    """
    if not _fonts:
        _initfonts ()
    if None in _fonts:
        return None
    return list (_families.keys ())

def find_font (name, bold=False, italic=False, ftype=None):
    """find_font (name, bold=False, italic=False, ftype=None) -> str, bool, bool

    Finds a font matching a certain family or font filename.

    Tries to find a font that matches the passed requirements best. The
    *name* argument denotes a specific font or font family name. If
    multiple fonts match that name, the *bold* and *italic* arguments
    are used to find a font that matches the requirements best. *ftype*
    is an optional font filetype argument to request specific font file
    types, such as bdf or ttf fonts.
    """
    if not _fonts:
        _initfonts ()
    if None in _fonts:
        return None

    if ftype:
        ftype = ftype.lower ()

    candidates = []
    #
    # font styles consist of (fullname, filetype, bold, italic)
    #
    for fname in _families.get (name, []):
        fullname, filetype, fbold, fitalic = _fonts[fname]
        if ftype and ftype != filetype:
            # The user requires a certain font filetype.
            continue

        if bold == fbold and italic == fitalic:
            # Exact style match
            candidates.append ((fname, fbold, fitalic, 0))
        elif italic and italic == fitalic:
            # Italic matches
            candidates.append ((fname, fbold, fitalic, 1))
        elif bold and bold == fbold:
            # Bold matches
            candidates.append ((fname, fbold, fitalic, 2))
        else:
            # None matches
            candidates.append ((fname, fbold, fitalic, 3))
    if candidates:
        candidates.sort(key=lambda x: x[3])
        return candidates[0][0], candidates[0][1], candidates[0][2]

    for items in _fonts.items():
        fname = items[0]
        fullname, filetype, fbold, fitalic = items[1]
        if fullname != name and name not in fname:
            nl = name.lower ()
            fl = fullname.lower ()
            if nl not in fl and nl not in fname.lower():
                continue
        if ftype and ftype != filetype:
            # The user requires a certain font filetype.
            continue

        if bold == fbold and italic == fitalic:
            # Exact style match
            candidates.append ((fname, fbold, fitalic, 0))
        elif italic and italic == fitalic:
            # Italic matches
            candidates.append ((fname, fbold, fitalic, 1))
        elif bold and bold == fbold:
            # Bold matches
            candidates.append ((fname, fbold, fitalic, 2))
        else:
            # None matches
            candidates.append ((fname, fbold, fitalic, 3))
    if candidates:
        candidates.sort(key=lambda x: x[3])
        return candidates[0][0], candidates[0][1], candidates[0][2]
    return None
