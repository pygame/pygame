import objc
from Foundation import NSObject, NSLog, NSBundle, NSDictionary
from AppKit import NSApplicationDelegate, NSTerminateLater, NSApplication, NSCriticalRequest, NSImage, NSApp, NSMenu, NSMenuItem
import os, sys
import pygame

# Make a good guess at the name of the application
if len(sys.argv) > 1:
    MyAppName = os.path.splitext(sys.argv[1])[0]
else:
    MyAppname = 'pygame'
    
# Need to do this if not running with a nib
def setupAppleMenu():
    appleMenuController = objc.lookUpClass('NSAppleMenuController').alloc().init()
    appleMenu = NSMenu.alloc().initWithTitle_('')
    appleMenuItem = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_('', None, '')
    appleMenuItem.setSubmenu_(appleMenu)
    NSApp().mainMenu().addItem_(appleMenuItem)
    appleMenuController.controlMenu_(appleMenu)
    NSApp().mainMenu().removeItem_(appleMenuItem)
    
# Need to do this if not running with a nib
def setupWindowMenu():
    windowMenu = NSMenu.alloc().initWithTitle_('Window')
    menuItem = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_('Minimize', 'performMiniaturize:', 'm')
    windowMenu.addItem_(menuItem)
    del menuItem
    windowMenuItem = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_('Window', None, '')
    windowMenuItem.setSubmenu_(windowMenu)
    NSApp().mainMenu().addItem_(windowMenuItem)
    NSApp().setWindowsMenu_(windowMenu)

# Used to cleanly terminate
class MyAppDelegate(NSObject, NSApplicationDelegate):
    def init(self):
        return self

    def applicationDidFinishLaunching_(self, aNotification):
        pass

    def applicationShouldTerminate_(self, app):
        import pygame
        pygame.event.post(pygame.event.Event(pyame.QUIT))
        return NSTerminateLater

# Start it up!
app = NSApplication.sharedApplication()
defaultIcon = os.path.join(os.path.split(__file__)[0], 'pygame_icon.tiff')
if os.path.exists(defaultIcon):
    img = NSImage.alloc().initWithContentsOfFile_(defaultIcon)
    if img:
        app.setApplicationIconImage_(img)

DELEGATE = MyAppDelegate.alloc().init()
app.setDelegate_(DELEGATE)
if not app.mainMenu():
    mainMenu = NSMenu.alloc().init()
    app.setMainMenu_(mainMenu)
    setupAppleMenu()
    setupWindowMenu()
app.finishLaunching()
app.updateWindows()
app.activateIgnoringOtherApps_(objc.YES)
