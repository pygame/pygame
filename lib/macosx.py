from Foundation import *
from AppKit import *
import os, sys
import objc
import MacOS
from pygame.pkgdata import getResource
from pygame.base import get_sdl_version

__all__ = ['init']

# Need to do this if not running with a nib
def setupAppleMenu(app):
    appleMenuController = NSAppleMenuController.alloc().init()
    appleMenuController.retain()
    appleMenu = NSMenu.alloc().initWithTitle_('')
    appleMenuItem = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_('', None, '')
    appleMenuItem.setSubmenu_(appleMenu)
    app.mainMenu().addItem_(appleMenuItem)
    appleMenuController.controlMenu_(appleMenu)
    app.mainMenu().removeItem_(appleMenuItem)
    
# Need to do this if not running with a nib
def setupWindowMenu(app):
    windowMenu = NSMenu.alloc().initWithTitle_('Window')
    windowMenu.retain()
    menuItem = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_('Minimize', 'performMiniaturize:', 'm')
    windowMenu.addItem_(menuItem)
    windowMenuItem = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_('Window', None, '')
    windowMenuItem.setSubmenu_(windowMenu)
    app.mainMenu().addItem_(windowMenuItem)
    app.setWindowsMenu_(windowMenu)

# Used to cleanly terminate
class PyGameAppDelegate(NSObject):
    def applicationShouldTerminate_(self, app):
        import pygame.event
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        return NSTerminateLater

    def windowUpdateNotification_(self, notification):
        win = notification.object()
        if get_sdl_version() < (1, 2, 8) and isinstance(win, objc.lookUpClass('SDL_QuartzWindow')):
            # Seems to be a retain count bug in SDL.. workaround!
            win.retain()
        NSNotificationCenter.defaultCenter().removeObserver_name_object_(
            self, NSWindowDidUpdateNotification, None)
        self.release()

def setIcon(app):
    try:
        defaultIcon = getResource('pygame_icon.tiff').read()
    except IOError:
        return
    data = NSData.dataWithBytes_length_(defaultIcon, len(defaultIcon))
    if data is None:
        return
    img = NSImage.alloc().initWithData_(data)
    if img is None:
        return
    app.setApplicationIconImage_(img)

def install():
    app = NSApplication.sharedApplication()
    setIcon(app)
    appDelegate = PyGameAppDelegate.alloc().init()
    app.setDelegate_(appDelegate)
    appDelegate.retain()
    NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
        appDelegate,
        'windowUpdateNotification:',
        NSWindowDidUpdateNotification,
        None)
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
    app = NSApplication.sharedApplication()
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
        os.chdir(os.path.dirname(sys.argv[0]))
    return True
