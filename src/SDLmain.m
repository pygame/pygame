/*
       Initial Version: Darrell Walisser <dwaliss1@purdue.edu>
       Non-NIB-Code & other changes: Max Horn <max@quendi.de>
       Hacked for pygame: Bob Ippolito <bob@redivi.com>

       Feel free to customize this file to suit your needs
*/

#import "Python.h"

#import "SDL.h"
#import <sys/param.h>
#import <unistd.h>
#import <Cocoa/Cocoa.h>
#import <Carbon/Carbon.h>
#import "CPS.h"

@interface SDLApplication : NSApplication {}
- (void)quit:(id)sender;
- (void)terminate:(id)sender;
@end

@interface SDLMain : NSObject
@end

@interface NSAppleMenuController:NSObject {}
- (void)controlMenu:(NSMenu *)aMenu;
@end

void setupAppleMenu(void)
{
    /* warning: this code is very odd */
    NSAppleMenuController *appleMenuController;
    NSMenu *appleMenu;
    NSMenuItem *appleMenuItem;

    appleMenuController = [[NSAppleMenuController alloc] init];
    appleMenu = [[NSMenu alloc] initWithTitle:@""];
    appleMenuItem = [[NSMenuItem alloc] initWithTitle:@"" action:nil keyEquivalent:@""];
    
    [appleMenuItem setSubmenu:appleMenu];

    /* yes, we do need to add it and then remove it --
       if you don't add it, it doesn't get displayed
       if you don't remove it, you have an extra, titleless item in the menubar
       when you remove it, it appears to stick around
       very, very odd */
    [[NSApp mainMenu] addItem:appleMenuItem];
    [appleMenuController controlMenu:appleMenu];
    [[NSApp mainMenu] removeItem:appleMenuItem];
    [appleMenu release];
    [appleMenuItem release];
}

/* Create a window menu */
void setupWindowMenu(void)
{
    NSMenu	*windowMenu;
    NSMenuItem	*windowMenuItem;
    NSMenuItem	*menuItem;


    windowMenu = [[NSMenu alloc] initWithTitle:@"Window"];
    
    /* "Minimize" item */
    menuItem = [[NSMenuItem alloc] initWithTitle:@"Minimize" action:@selector(performMiniaturize:) keyEquivalent:@"m"];
    [windowMenu addItem:menuItem];
    [menuItem release];
    
    /* Put menu into the menubar */
    windowMenuItem = [[NSMenuItem alloc] initWithTitle:@"Window" action:nil keyEquivalent:@""];
    [windowMenuItem setSubmenu:windowMenu];
    [[NSApp mainMenu] addItem:windowMenuItem];
    
    /* Tell the application object that this is now the window menu */
    [NSApp setWindowsMenu:windowMenu];

    /* Finally give up our references to the objects */
    [windowMenu release];
    [windowMenuItem release];
}

/* The main class of the application, the application's delegate */
@implementation SDLApplication : NSApplication
- (void)quit:(id)sender
{
    /* Post a SDL_QUIT event */
    SDL_Event event;
    event.type = SDL_QUIT;
    SDL_PushEvent(&event);
}

- (void)terminate:(id)sender
{
    /* Post a SDL_QUIT event */
    SDL_Event event;
    event.type = SDL_QUIT;
    SDL_PushEvent(&event);
}
@end

@implementation SDLMain

NSAutoreleasePool *global_pool;
SDLMain *sdlMain;

void StartTheApplication (void)
{
    //OSErr err;
    CPSProcessSerNum PSN;
    NSImage *pygameIcon;
    global_pool = [[NSAutoreleasePool alloc] init];
    char* pygame_icon_path = NULL;

    /*get the path to the icon*/

    PyObject* init_module = PyImport_ImportModule("pygame");
    if (!init_module)
        PyErr_Clear();
    else 
    {
        char* path = PyModule_GetFilename(init_module);
	if(!path)
	    PyErr_Clear();
	else 
	{
	    char* endp = strstr(path, "__init__.");
	    if(endp) 
	    {
	        pygame_icon_path = PyMem_Malloc(strlen(path)+20);
		if(pygame_icon_path) 
		{
		    strncpy(pygame_icon_path, path, endp-path);
		    strcpy(pygame_icon_path+(endp-path), "pygame_icon.tiff");
		}
	    }
	}
    }

    /*ensure application object is initialized*/
    [SDLApplication sharedApplication];
    
    /*tell the dock about us*/
    if (!CPSGetCurrentProcess(&PSN))
        if (!CPSSetProcessName(&PSN,"pygame"))
            if (!CPSEnableForegroundOperation(&PSN,0x03,0x3C,0x2C,0x1103))
                if (!CPSSetFrontProcess(&PSN))
                    [NSApplication sharedApplication];
		    
    /*setup menubar*/
    [NSApp setMainMenu:[[NSMenu alloc] init]];
    setupAppleMenu();
    setupWindowMenu();
    
    /*create app and delegate*/
    sdlMain = [[SDLMain alloc] init];
    [NSApp setDelegate:sdlMain];
    [NSApp finishLaunching];
    [NSApp requestUserAttention:NSCriticalRequest];
    [NSApp updateWindows];


    /*set icon*/
    pygameIcon = [[NSImage alloc] initWithContentsOfFile: [NSString stringWithCString: pygame_icon_path]];
    [NSApp setApplicationIconImage:pygameIcon];

}

void WeAreDoneFreeSomeMemory(void)
{
    [sdlMain release];
    [global_pool release];
}

@end
