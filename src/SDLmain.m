/*
       Initial Version: Darrell Walisser <dwaliss1@purdue.edu>
       Non-NIB-Code & other changes: Max Horn <max@quendi.de>
       Hacked for pygame: Bob Ippolito <bob@redivi.com>

       Feel free to customize this file to suit your needs
*/

#import "SDL.h"
#import <sys/param.h>
#import <unistd.h>
#import <Cocoa/Cocoa.h>
#import <Carbon/Carbon.h>
#import "CPS.h"

@interface SDLMain : NSObject
@end

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
@implementation SDLMain
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


NSAutoreleasePool *global_pool;
SDLMain *sdlMain;

void StartTheDamnApplication (void)
{
    CPSProcessSerNum PSN;
    //OSErr err;
    NSImage *pygameIcon;
    global_pool = [[NSAutoreleasePool alloc] init];
    [NSApplication sharedApplication];
    if (!CPSGetCurrentProcess(&PSN))
        if (!CPSSetProcessName(&PSN,"pygame"))
            if (!CPSEnableForegroundOperation(&PSN,0x03,0x3C,0x2C,0x1103))
                if (!CPSSetFrontProcess(&PSN))
                    [NSApplication sharedApplication];
    [NSApp setMainMenu:[[NSMenu alloc] init]];
    setupWindowMenu();
    sdlMain = [[SDLMain alloc] init];
    [NSApp setDelegate:sdlMain];
    [NSApp finishLaunching];
    [NSApp requestUserAttention:NSCriticalRequest];
    [NSApp updateWindows];
    pygameIcon = [[NSImage alloc] initWithContentsOfFile: @"/Library/Frameworks/Python.framework/Versions/Current/lib/python2.2/site-packages/pygame/pygame_icon.tiff"];
    [NSApp setApplicationIconImage:pygameIcon];

}

void WeAreDoneFreeSomeMemory(void)
{
    [sdlMain release];
    [global_pool release];
}

@end