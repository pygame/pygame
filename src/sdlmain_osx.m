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

struct CPSProcessSerNum
{
	UInt32 lo;
	UInt32 hi;
};
typedef struct CPSProcessSerNum CPSProcessSerNum;

extern OSErr CPSGetCurrentProcess( CPSProcessSerNum *psn);
extern OSErr CPSEnableForegroundOperation( CPSProcessSerNum *psn, UInt32 _arg2, UInt32 _arg3, UInt32 _arg4, UInt32 _arg5);
extern OSErr CPSSetFrontProcess( CPSProcessSerNum *psn);
extern OSErr CPSSetProcessName ( CPSProcessSerNum *psn, const char *processname );

static bool HasInstalledApplication = 0;

static NSString *getApplicationName(void)
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

static PyObject*
_WMEnable(PyObject* self)
{
	CPSProcessSerNum psn;
    OSErr err;
    const char* nameString;
    NSString* nameNSString;
    
    err = CPSGetCurrentProcess(&psn);
    if (err == 0)
    {
    	nameNSString = getApplicationName();
    	nameString = [nameNSString UTF8String];
    	CPSSetProcessName(&psn, nameString);

        err = CPSEnableForegroundOperation(&psn,0x03,0x3C,0x2C,0x1103);
        if (err == 0)
        {
        	err = CPSSetFrontProcess(&psn);
        	if (err != 0)
        	{
            	return RAISE (PyExc_SDLError, "CPSSetFrontProcess failed");        		
        	}
        }
        else
        {
        	return RAISE (PyExc_SDLError, "CPSEnableForegroundOperation failed");
        }
    }
    else
    {
    	return RAISE (PyExc_SDLError, "CPSGetCurrentProcess failed");
    }
    
    Py_RETURN_TRUE;
}

static PyObject*
_RunningFromBundleWithNSApplication(PyObject* self)
{
	if (HasInstalledApplication)
	{
		Py_RETURN_TRUE;
	}
	CFBundleRef MainBundle = CFBundleGetMainBundle();
	if (MainBundle != NULL)
	{
		if (CFBundleGetDataPointerForName(MainBundle, CFSTR("NSApp")) != NULL)
		{
			Py_RETURN_TRUE;
		}
	}
    Py_RETURN_FALSE;
}

//#############################################################################
// Defining the NSApplication class we will use
//#############################################################################
@interface SDLApplication : NSApplication
@end

/* For some reaon, Apple removed setAppleMenu from the headers in 10.4,
 but the method still is there and works. To avoid warnings, we declare
 it ourselves here. */
@interface NSApplication(SDL_Missing_Methods)
- (void)setAppleMenu:(NSMenu *)menu;
@end

@implementation SDLApplication
/* Invoked from the Quit menu item */
- (void)terminate:(id)sender
{
    SDL_Event event;
    event.type = SDL_QUIT;
    SDL_PushEvent(&event);
}
@end

@interface SDLApplicationDelegate : NSObject
@end
@implementation SDLApplicationDelegate
- (BOOL)application:(NSApplication *)theApplication openFile:(NSString *)filename
{
    int posted;

    /* Post the event, if desired */
    posted = 0;
	SDL_Event event;
	event.type = SDL_USEREVENT;
	event.user.code = 0x1000;
	event.user.data1 = SDL_strdup([filename UTF8String]);
	posted = (SDL_PushEvent(&event) > 0);
    return (BOOL)(posted);
}
@end

static void setApplicationMenu(void)
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
    [menuItem setKeyEquivalentModifierMask:(NSAlternateKeyMask|NSCommandKeyMask)];

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

static void setupWindowMenu(void)
{
    NSMenu      *windowMenu;
    NSMenuItem  *windowMenuItem;
    NSMenuItem  *menuItem;

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

static PyObject*
_InstallNSApplication(PyObject* self, PyObject* arg)
{
    char* icon_data = NULL;
    int data_len = 0;
    SDLApplicationDelegate *sdlApplicationDelegate = NULL;

    NSAutoreleasePool	*pool = [[NSAutoreleasePool alloc] init];

    [SDLApplication sharedApplication];

    if (PyArg_ParseTuple (arg, "|z#", &icon_data, &data_len))
    {
        NSData *image_data = [NSData dataWithBytes:icon_data length:data_len];
	    NSImage *icon_img = [[NSImage alloc] initWithData:image_data];
	    if (icon_img != NULL)
	    {
	    	[NSApp setApplicationIconImage:icon_img];
	    }
    }

    [NSApp setMainMenu:[[NSMenu alloc] init]];
    setApplicationMenu();
    setupWindowMenu();

    [NSApp finishLaunching];
    [NSApp updateWindows];
    [NSApp activateIgnoringOtherApps:true];

    HasInstalledApplication = 1;

    /* Create SDLApplicationDelegate and make it the app delegate */
    sdlApplicationDelegate = [[SDLApplicationDelegate alloc] init];
    [NSApp setDelegate:sdlApplicationDelegate];
    
	Py_RETURN_TRUE;
}

static PyMethodDef macosx_builtins[] =
{
    { "WMEnable", (PyCFunction) _WMEnable, METH_NOARGS, "Enables Foreground Operation when Window Manager is not available" },
    { "RunningFromBundleWithNSApplication", (PyCFunction) _RunningFromBundleWithNSApplication, METH_NOARGS, "Returns true if we are running from an AppBundle with a variable named NSApp" },
    { "InstallNSApplication", _InstallNSApplication, METH_VARARGS, "Creates an NSApplication with the right behaviors for SDL" },
    { NULL, NULL, 0, NULL}
};



MODINIT_DEFINE (sdlmain_osx)
{
    PyObject *module;

    /* create the module */

#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        MODPREFIX "sdlmain_osx",
        NULL,
        -1,
        macosx_builtins,
        NULL, NULL, NULL, NULL
    };
#endif


#if PY3
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 (MODPREFIX "sdlmain_osx", macosx_builtins, NULL);
#endif


    /*imported needed apis*/
    import_pygame_base ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }


    MODINIT_RETURN (module);
}
