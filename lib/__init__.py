##    pygame - Python Game Library
##    Copyright (C) 2000-2001  Pete Shinners
##
##    This library is free software; you can redistribute it and/or
##    modify it under the terms of the GNU Library General Public
##    License as published by the Free Software Foundation; either
##    version 2 of the License, or (at your option) any later version.
##
##    This library is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
##    Library General Public License for more details.
##
##    You should have received a copy of the GNU Library General Public
##    License along with this library; if not, write to the Free
##    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##
##    Pete Shinners
##    pete@shinners.org

# main pygame source
# lets get things coordinated

def import_all_pygame():
    import sys
    if sys.platform=='darwin':
        import os
        # this may change someday, but we want to chdir to where our file is if we're in / for no
        # good reason..
        if (os.getcwd() == '/') and len(sys.argv):
            os.chdir(os.path.split(sys.argv[0])[0])
        else:
            argv0=''
            if len(sys.argv): argv0=sys.argv[0]
            print "WARNING!  Running pygame apps from any method other than through python.app (aka through the finder or launchservices) is UNSUPPORTED!"
            print "          If you insist on using the terminal, type \"open %s\", and hold down the option key if you need to" % (argv0)
            print "          specify additional command line arguments.  A dialog box will pop up and make you happy, I promise."
            print ""
            print "          I sure hope you ran as \"%s %s\" exactly, otherwise you will really have problems."%(sys.executable,' '.join(sys.argv))
            print "          WindowServer doesn't like what you're doing as is, and it gets really funky if you run things from the path for whatever reason."
            print ""
            # not ready for prime time yet, it just rewrites the commandline so windowserver can pick it up
            #import pygame.macosx

    def makemodules(**mods): return mods
    modules = makemodules(
        base=2, cdrom=0, constants=2, cursors=0, display=0,
        draw=0, event=0, font=0, image=0, joystick=0,
        key=0, mixer=0, mouse=0, movie=0, rect=1, sprite=0,
        surface=0, time=0, transform=0, surfarray=0, version=2
    )
    for mod, required in modules.items():
        try:
            module = __import__('pygame.'+mod, None, None, ['pygame.'+mod])
            globals()[mod] = module
            if required == 2:
                for item in dir(module):
                    if item[0] != '_':
                        globals()[item] = getattr(module, item)
        except ImportError:
            if required:
                CannotImportPygame = ImportError
                MissingModule =  "Cannot Import 'pygame.%s'"%mod
                raise CannotImportPygame, MissingModule
            globals()[mod] = None

import_all_pygame()
del import_all_pygame

Surface = getattr(surface, 'Surface', lambda:Missing_Pygame_Function)
Rect = getattr(rect, 'Rect', lambda:Missing_Pygame_Function)


