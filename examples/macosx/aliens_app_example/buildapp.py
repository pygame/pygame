from bundlebuilder2 import buildapp
from plistlib import Plist, Dict
import glob, imp, shutil, os

NAME = 'aliens'
VERSION = '0.1'

infoPlist = Plist(
    CFBundleIconFile            = NAME,
    CFBundleName                = NAME,
    CFBundleShortVersionString  = VERSION,
    CFBundleGetInfoString       = ' '.join([NAME, VERSION]),
    CFBundleExecutable          = NAME,
)

# pkgdata stuff
pygamedata = os.path.join('pkgdata', 'pygame')
if not os.path.exists(pygamedata):
    os.makedirs(pygamedata)
    pygamedir = imp.find_module('pygame')[1]
    for fn in glob.glob(os.path.join(pygamedir, '*.ttf')):
        shutil.copy(fn, pygamedata)

buildapp(
    name        = NAME,
    bundle_id   = 'org.pygame.aliens',
    mainprogram = 'aliens_bootstrap.py',
    nibname     = 'MainMenu',
    resources   = ["English.lproj", "../../data", "pkgdata"],
    plist       = infoPlist,
)
