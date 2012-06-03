#################################### IMPORTS ###################################

if __name__ == '__main__':
    import sys
    import os
    pkg_dir = os.path.split(os.path.abspath(__file__))[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

if is_pygame_pkg:
    from pygame.tests.test_utils import test_not_implemented, unittest
else:
    from test.test_utils import test_not_implemented, unittest
import pygame

################################################################################

class BufferProxyTypeTest(unittest.TestCase):
    def test_creation(self):
        self.assert_(pygame.bufferproxy.BufferProxy())

    def todo_test_length(self):

        # __doc__ (as of 2008-08-02) for pygame.bufferproxy.BufferProxy.length:

          # The size of the buffer data in bytes.

        self.fail() 

    def todo_test_raw(self):

        # __doc__ (as of 2008-08-02) for pygame.bufferproxy.BufferProxy.raw:

          # The raw buffer data as string. The string may contain NUL bytes.

        self.fail() 

    def todo_test_write(self):

        # __doc__ (as of 2008-08-02) for pygame.bufferproxy.BufferProxy.write:

          # B.write (bufferproxy, buffer, offset) -> None
          # 
          # Writes raw data to the bufferproxy.
          # 
          # Writes the raw data from buffer to the BufferProxy object, starting
          # at the specified offset within the BufferProxy.
          # If the length of the passed buffer exceeds the length of the
          # BufferProxy (reduced by the offset), an IndexError will be raised.

        self.fail() 
        
################################################################################

if __name__ == '__main__':
    unittest.main()
