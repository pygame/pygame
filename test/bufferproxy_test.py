#################################### IMPORTS ###################################

import test_utils
import test.unittest as unittest
from test_utils import test_not_implemented

################################################################################

class BufferProxyTypeTest(unittest.TestCase):
    def test_write(self):

        # __doc__ (as of 2008-06-25) for pygame.bufferproxy.BufferProxy.write:

          # B.write (bufferproxy, buffer, offset) -> None
          # 
          # Writes raw data to the bufferproxy.
          # 
          # Writes the raw data from buffer to the BufferProxy object, starting
          # at the specified offset within the BufferProxy.
          # If the length of the passed buffer exceeds the length of the
          # BufferProxy (reduced by the offset), an IndexError will be raised.

        self.assert_(test_not_implemented()) 


################################################################################

if __name__ == '__main__':
    unittest.main()
