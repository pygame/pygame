######################### PYGAME UNITTEST STUBBER TESTS ########################
"""

TODO
====

more, more, more

"""

import pygame, unittest
pygame.init()

from gen_stubs import get_package_modules, module_test_stubs

class TestStubGenerator(unittest.TestCase):
    def test_get_package_modules(self):
        " get_package_modules(pygame): each m must have 'pygame' in its repr()"
        # eg pygame.mixer, pygame.color
        
        # There should be none where 'pygame' is not in the name of the module        
        self.assert_(not filter( lambda t: 'pygame' not in repr(t),
                                 get_package_modules(pygame) ) 
        )

if __name__ == "__main__":
    if 1:
        unittest.main()
    else:
        # scribble tests -->
        
        for _, stub in module_test_stubs(pygame):
            print stub
        
################################################################################