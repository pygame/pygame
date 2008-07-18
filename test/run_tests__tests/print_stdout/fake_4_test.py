import test_utils
import test.unittest as unittest

class KeyModuleTest(unittest.TestCase):
    def test_get_focused(self):
        self.assert_(True) 

    def test_get_mods(self):
        self.assert_(True) 

    def test_get_pressed(self):
        self.assert_(False, "Some Jibberish") 

    def test_name(self):
        self.assert_(True) 

    def test_set_mods(self):
        if 1:
            if 1:
                assert False 

    def test_set_repeat(self):
        self.assert_(True) 

if __name__ == '__main__':
    unittest.main()