from Foundation import *
from AppKit import *
import os, sys
import objc
import MacOS
from pygame.pkgdata import getResourcePath

__all__ = ['init']

# Need to do this if not running with a nib
def setupAppleMenu(app):
    appleMenuController = NSAppleMenuController.alloc().init()
    appleMenu = NSMenu.alloc().initWithTitle_('')
    appleMenuItem = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_('', None, '')
    appleMenuItem.setSubmenu_(appleMenu)
    app.mainMenu().addItem_(appleMenuItem)
    appleMenuController.controlMenu_(appleMenu)
    app.mainMenu().removeItem_(appleMenuItem)
    
# Need to do this if not running with a nib
def setupWindowMenu(app):
    windowMenu = NSMenu.alloc().initWithTitle_('Window')
    menuItem = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_('Minimize', 'performMiniaturize:', 'm')
    windowMenu.addItem_(menuItem)
    windowMenuItem = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_('Window', None, '')
    windowMenuItem.setSubmenu_(windowMenu)
    app.mainMenu().addItem_(windowMenuItem)
    app.setWindowsMenu_(windowMenu)

# Used to cleanly terminate
class PyGameAppDelegate(NSObject, NSApplicationDelegate):
    def applicationShouldTerminate_(self, app):
        import pygame.event
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        return NSTerminateLater

def setIcon(app):
    try:
        defaultIcon = getResourcePath('pygame_icon.tiff')
    except IOError:
        pass
    else:
        img = NSImage.alloc().initWithContentsOfFile_(defaultIcon)
        if img:
            app.setApplicationIconImage_(img)

def install():
    global _applicationDelegate
    app = NSApplication.sharedApplication()
    setIcon(app)
    _applicationDelegate = PyGameAppDelegate.alloc().init()
    app.setDelegate_(_applicationDelegate)
    if not app.mainMenu():
        mainMenu = NSMenu.alloc().init()
        app.setMainMenu_(mainMenu)
        setupAppleMenu(app)
        setupWindowMenu(app)
    app.finishLaunching()
    app.updateWindows()
    app.activateIgnoringOtherApps_(True)

def S(*args):
    return ''.join(args)

OSErr = objc._C_SHT
OUTPSN = 'o^{ProcessSerialNumber=LL}'
INPSN = 'n^{ProcessSerialNumber=LL}'

FUNCTIONS=[
    # These two are public API
    ( u'GetCurrentProcess', S(OSErr, OUTPSN) ),
    ( u'SetFrontProcess', S(OSErr, INPSN) ),
    # This is undocumented SPI
    ( u'CPSSetProcessName', S(OSErr, INPSN, objc._C_CHARPTR) ),
    ( u'CPSEnableForegroundOperation', S(OSErr, INPSN) ),
]

def WMEnable(name=None):
    if name is None:
        name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    if isinstance(name, unicode):
        name = name.encode('utf-8')
    if not hasattr(objc, 'loadBundleFunctions'):
        return False
    bndl = NSBundle.bundleWithPath_(objc.pathForFramework('/System/Library/Frameworks/ApplicationServices.framework'))
    if bndl is None:
        print >>sys.stderr, 'ApplicationServices missing'
        return False
    d = {}
    objc.loadBundleFunctions(bndl, d, FUNCTIONS)
    for (fn, sig) in FUNCTIONS:
        if fn not in d:
            print >>sys.stderr, 'Missing', fn
            return False
    err, psn = d['GetCurrentProcess']()
    if err:
        print >>sys.stderr, 'GetCurrentProcess', (err, psn)
        return False
    err = d['CPSSetProcessName'](psn, name)
    if err:
        print >>sys.stderr, 'CPSSetProcessName', (err, psn)
        return False
    err = d['CPSEnableForegroundOperation'](psn)
    if err:
        print >>sys.stderr, 'CPSEnableForegroundOperation', (err, psn)
        return False
    err = d['SetFrontProcess'](psn)
    if err:
        print >>sys.stderr, 'SetFrontProcess', (err, psn)
        return False
    return True

def init():
    if not (MacOS.WMAvailable() or WMEnable()):
        raise ImportError, "Can not access the window manager.  Use py2app or execute with the pythonw script."
    if not NSApp():
        # running outside of a bundle
        install()
    # running inside a bundle, change dir
    if (os.getcwd() == '/') and len(sys.argv) > 1:
        os.chdir(os.path.basedir(sys.argv[0]))
    return True
