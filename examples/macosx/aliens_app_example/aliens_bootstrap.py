from Foundation import NSBundle
from AppKit import NSTerminateNow, NSApp, NSRunAlertPanel
from PyObjCTools import NibClassBuilder, AppHelper

def exception_handler():
    import traceback, sys, os
    typ, info, trace = sys.exc_info()
    if typ in (KeyboardInterrupt, SystemExit):
        return
    tracetop = traceback.extract_tb(trace)[-1]
    tracetext = 'File %s, Line %d' % tracetop[:2]
    if tracetop[2] != '?':
        tracetext += ', Function %s' % tracetop[2]
    exception_message = '%s:\n%s\n\n%s\n"%s"'
    message = exception_message % (str(type), str(info), tracetext, tracetop[3])
    title = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    title = title.capitalize() + ' Error'
    NSRunAlertPanel(title, message, None, None, None)

NibClassBuilder.extractClasses("MainMenu")
class PygameAppDelegate(NibClassBuilder.AutoBaseClass):
    def applicationDidFinishLaunching_(self, aNotification):
        try:
            import aliens
            aliens.main()
        except:
            exception_handler()
        NSApp().terminate_(self)

    def applicationShouldTerminate_(self, app):
        import pygame, pygame.event
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        return NSTerminateNow

if __name__ == '__main__':
    AppHelper.runEventLoop()
