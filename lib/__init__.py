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
    def makemodules(**mods): return mods
    modules = makemodules(
        base=2, cdrom=0, constants=2, cursors=0, display=0,
        draw=0, event=0, font=0, image=0, joystick=0,
        key=0, mixer=0, mouse=0, movie=0, rect=1, sprite=0,
        surface=0, time=0, transform=0, surfarray=0
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


