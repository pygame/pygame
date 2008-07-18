import test_utils
import test.unittest as unittest
import sys

class KeyModuleTest(unittest.TestCase):
    def test_get_focused(self):
        self.assert_(True) 

    def test_get_mods(self):
        self.assert_(True) 

    def test_get_pressed(self):
        self.assert_(True) 

    def test_name(self):
        print >> sys.stderr, 'jibberish messes things up'
        self.assert_(False)

    def test_set_mods(self):
        self.assert_(True) 

    def test_set_repeat(self):
        self.assert_(True) 

if __name__ == '__main__':
    unittest.main()