from Foundation import NSObject, NSLog, NSBundle, NSDictionary
from AppKit import NSAppleMenuController, NSApplicationDelegate, NSTerminateLater, NSApplication, NSImage, NSMenu, NSMenuItem
import os, sys
import pygame
from pygame.pkgdata import getResourcePath

__all__ = ['install']

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
