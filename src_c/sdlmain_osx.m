/*
    pygame - Python Game Library
    Copyright (C) 2009 Brian Fisher

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public
    License along with this library; if not, write to the Free
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

/* Mac OS X functions to accommodate the fact SDLMain.m is not included */

#include "pygame.h"

#include <Carbon/Carbon.h>
#include <Foundation/Foundation.h>
#include <AppKit/NSApplication.h>
#include <AppKit/NSMenuItem.h>
#include <AppKit/NSMenu.h>
#include <AppKit/NSEvent.h>
#include <Foundation/NSData.h>
#include <AppKit/NSImage.h>

#include "pgcompat.h"
#include "scrap.h"

#include <AvailabilityMacros.h>
/* We dont support OSX 10.6 and below. */
#if __MAC_OS_X_VERSION_MAX_ALLOWED <= 1060
    #define PYGAME_MAC_SCRAP_OLD 1
#endif

struct CPSProcessSerNum {
    UInt32 lo;
    UInt32 hi;
};

typedef struct CPSProcessSerNum CPSProcessSerNum;

extern OSErr CPSGetCurrentProcess(CPSProcessSerNum *psn);
extern OSErr CPSEnableForegroundOperation(CPSProcessSerNum *psn, UInt32 _arg2, UInt32 _arg3, UInt32 _arg4, UInt32 _arg5);
extern OSErr CPSSetFrontProcess(CPSProcessSerNum *psn);
extern OSErr CPSSetProcessName(CPSProcessSerNum *psn, const char *processname);

static NSString *
getApplicationName(void)
{
    const NSDictionary *dict;
    NSString *appName = 0;

    /* Determine the application name */
    dict = (const NSDictionary *)CFBundleGetInfoDictionary(CFBundleGetMainBundle());
    if (dict)
        appName = [dict objectForKey: @"CFBundleName"];

    if (![appName length])
        appName = [[NSProcessInfo processInfo] processName];

    return appName;
}

static PyObject *
_WMEnable(PyObject *self)
{
    CPSProcessSerNum psn;
    const char* nameString;
    NSString* nameNSString;

    if (!CPSGetCurrentProcess(&psn)) {
        nameNSString = getApplicationName();
        nameString = [nameNSString UTF8String];
        CPSSetProcessName(&psn, nameString);

        if (!CPSEnableForegroundOperation(&psn, 0x03, 0x3C, 0x2C, 0x1103)) {
            if (CPSSetFrontProcess(&psn))
                return RAISE(pgExc_SDLError, "CPSSetFrontProcess failed");
        }
        else
            return RAISE(pgExc_SDLError, "CPSEnableForegroundOperation failed");
    }
    else
        return RAISE(pgExc_SDLError, "CPSGetCurrentProcess failed");

    Py_RETURN_TRUE;
}

//#############################################################################
// Defining the NSApplication class we will use
//#############################################################################
@interface PYGSDLApplication : NSApplication
@end

/* For some reason, Apple removed setAppleMenu from the headers in 10.4,
 but the method still is there and works. To avoid warnings, we declare
 it ourselves here. */
@interface NSApplication(SDL_Missing_Methods)
- (void)setAppleMenu:(NSMenu *)menu;
@end

@implementation PYGSDLApplication
/* Invoked from the Quit menu item */
- (void)terminate:(id)sender
{
    SDL_Event event;
    event.type = SDL_QUIT;
    SDL_PushEvent(&event);
}
@end

/* The below functions are unused for now, hence commented
static void
setApplicationMenu(void)
{
    NSMenu *appleMenu;
    NSMenuItem *menuItem;
    NSString *title;
    NSString *appName;

    appName = getApplicationName();
    appleMenu = [[NSMenu alloc] initWithTitle:@""];


    title = [@"About " stringByAppendingString:appName];
    [appleMenu addItemWithTitle:title action:@selector(orderFrontStandardAboutPanel:) keyEquivalent:@""];

    [appleMenu addItem:[NSMenuItem separatorItem]];

    title = [@"Hide " stringByAppendingString:appName];
    [appleMenu addItemWithTitle:title action:@selector(hide:) keyEquivalent:@"h"];

    menuItem = (NSMenuItem *)[appleMenu addItemWithTitle:@"Hide Others" action:@selector(hideOtherApplications:) keyEquivalent:@"h"];

#if MAC_OS_X_VERSION_MAX_ALLOWED < 101200
    [menuItem setKeyEquivalentModifierMask:(NSAlternateKeyMask|NSCommandKeyMask)];
#else
    [menuItem setKeyEquivalentModifierMask:(NSEventModifierFlagOption|NSEventModifierFlagCommand)];
#endif


    [appleMenu addItemWithTitle:@"Show All" action:@selector(unhideAllApplications:) keyEquivalent:@""];

    [appleMenu addItem:[NSMenuItem separatorItem]];

    title = [@"Quit " stringByAppendingString:appName];
    [appleMenu addItemWithTitle:title action:@selector(terminate:) keyEquivalent:@"q"];


    menuItem = [[NSMenuItem alloc] initWithTitle:@"" action:nil keyEquivalent:@""];
    [menuItem setSubmenu:appleMenu];
    [[NSApp mainMenu] addItem:menuItem];

    [NSApp setAppleMenu:appleMenu];

    [appleMenu release];
    [menuItem release];
}

static void
setupWindowMenu(void)
{
    NSMenu *windowMenu;
    NSMenuItem *windowMenuItem, *menuItem;

    windowMenu = [[NSMenu alloc] initWithTitle:@"Window"];

    menuItem = [[NSMenuItem alloc] initWithTitle:@"Minimize" action:@selector(performMiniaturize:) keyEquivalent:@"m"];
    [windowMenu addItem:menuItem];
    [menuItem release];

    windowMenuItem = [[NSMenuItem alloc] initWithTitle:@"Window" action:nil keyEquivalent:@""];
    [windowMenuItem setSubmenu:windowMenu];
    [[NSApp mainMenu] addItem:windowMenuItem];

    [NSApp setWindowsMenu:windowMenu];

    [windowMenu release];
    [windowMenuItem release];
}
*/

static PyObject *
_ScrapInit(PyObject *self)
{
    Py_RETURN_TRUE;
}

static PyObject*
_ScrapGet(PyObject *self, PyObject *args)
{
#if defined (PYGAME_MAC_SCRAP_OLD)
    Py_RETURN_NONE;
#else
    char *scrap_type;
    if (!PyArg_ParseTuple(args, "s", &scrap_type))
        return NULL;

    // anything else than text is not implemented
    if (strcmp(scrap_type, PYGAME_SCRAP_TEXT))
        Py_RETURN_NONE;

    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    NSString *info = [[NSPasteboard generalPasteboard]stringForType:NSPasteboardTypeString];
    PyObject *ret = NULL;

    if (info != nil)
        ret = PyUnicode_FromString([info UTF8String]);

    [pool release];
    if (!ret)
        Py_RETURN_NONE;
    return ret;
#endif
}

static PyObject *
_ScrapGetTypes(PyObject *self)
{
#ifdef PYGAME_MAC_SCRAP_OLD
    Py_RETURN_NONE;
#else
    PyObject *list = PyList_New(0);
    if (!list)
        return NULL;

    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    NSPasteboard *pb = [NSPasteboard generalPasteboard];
    NSArray *types = [pb types];

    for (NSString *type in types) {
        if (PyList_Append(list, PyUnicode_FromString([type UTF8String]))) {
            Py_DECREF(list);
            [pool release];
            return NULL;
        }
    }
    [pool release];
    return list;
#endif
}

static PyObject *
_ScrapPut(PyObject *self, PyObject *args)
{
#ifndef PYGAME_MAC_SCRAP_OLD
    char *scrap_type, *data;

    if (!PyArg_ParseTuple(args, "ss", &scrap_type, &data))
        return NULL;

    // anything else than text is not implemented
    if (strcmp(scrap_type, PYGAME_SCRAP_TEXT))
        Py_RETURN_NONE;

    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    NSPasteboard *pb = [NSPasteboard generalPasteboard];
    NSString *ndata = [NSString stringWithCString:(char *)data encoding:NSUTF8StringEncoding];
    [pb declareTypes: [NSArray arrayWithObject:NSPasteboardTypeString] owner: pb];
    [pb setString:ndata forType: NSPasteboardTypeString];
    [pool release];
#endif
    Py_RETURN_NONE;
}

static PyObject*
_ScrapSetMode(PyObject *self, PyObject *args)
{
#ifndef PYGAME_MAC_SCRAP_OLD
    char *mode;
    if (!PyArg_ParseTuple(args, "s", &mode))
        return NULL;
    /* ankith26:
     * TODO - Someone who understands what's going on here, pls fill code
     * here. Im just doing cleanup, I dont understand this stuff */
#endif
    Py_RETURN_NONE;
}

static PyObject*
_ScrapContains(PyObject *self, PyObject *args)
{
#ifdef PYGAME_MAC_SCRAP_OLD
    Py_RETURN_NONE;
#else
    char *mode;
    PyObject *ret = Py_False;
    if (!PyArg_ParseTuple (args, "s", &mode))
        return NULL;

    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    NSPasteboard *pb = [NSPasteboard generalPasteboard];
    NSArray *types = [pb types];
    for (NSString *type in types) {
        if (strcmp([type UTF8String], mode) == 0)
            ret = Py_True;
    }
    [pool release];

    Py_INCREF(ret);
    return ret;
#endif
}

static PyObject *
_ScrapLost(PyObject *self)
{
#ifdef PYGAME_MAC_SCRAP_OLD
    Py_RETURN_NONE;
#else
    PyObject *ret;
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    NSArray *supportedTypes =
            [NSArray arrayWithObjects: NSPasteboardTypeString, nil];
    NSString *bestType = [[NSPasteboard generalPasteboard]
            availableTypeFromArray:supportedTypes];
    ret = (bestType == nil) ? Py_True : Py_False;
    [pool release];

    Py_INCREF(ret);
    return ret;
#endif
}

static PyMethodDef macosx_builtins[] =
{
    {"WMEnable", (PyCFunction)_WMEnable, METH_NOARGS, "Enables Foreground Operation when Window Manager is not available" },
    {"ScrapInit", (PyCFunction)_ScrapInit, METH_NOARGS, "Initialize scrap for osx" },
    {"ScrapGet", (PyCFunction)_ScrapGet, METH_VARARGS, "Get a element from the scrap for osx" },
    {"ScrapPut", (PyCFunction)_ScrapPut, METH_VARARGS, "Set a element from the scrap for osx" },
    {"ScrapGetTypes", (PyCFunction)_ScrapGetTypes, METH_NOARGS, "Get scrap types for osx" },
    {"ScrapSetMode", (PyCFunction)_ScrapSetMode, METH_VARARGS, "Set mode for osx scrap (not used)" },
    {"ScrapContains", (PyCFunction)_ScrapContains, METH_VARARGS, "Check if a type is allowed on osx scrap (not used)" },
    {"ScrapLost", (PyCFunction)_ScrapLost, METH_NOARGS, "Check if our type is lost from scrap for osx" },
    {NULL, NULL, 0, NULL}
};

MODINIT_DEFINE (sdlmain_osx)
{
    /* create the module */
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "sdlmain_osx",
        NULL,
        -1,
        macosx_builtins,
        NULL,
        NULL,
        NULL,
        NULL
    };

    /*imported needed apis*/
    import_pygame_base();
    if (PyErr_Occurred()) {
        return NULL;
    }

    return PyModule_Create(&_module);
}
