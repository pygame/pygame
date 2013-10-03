##    pygame - Python Game Library
##    Copyright (C) 2000-2003  Pete Shinners
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

"""Simply the current installed pygame version. The version information is
stored in the regular pygame module as 'pygame.ver'. Keeping the version
information also available in a separate module allows you to test the
pygame version without importing the main pygame module.

The python version information should always compare greater than any previous
releases. (hmm, until we get to versions > 10)
"""

ver =   '1.9.2a0'
vernum = 1,9,2
