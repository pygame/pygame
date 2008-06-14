import unittest, os, test_utils
import re

class GL_ImageSave(unittest.TestCase):
    def test_image_save_works_with_opengl_surfaces(self):
        cmd = 'python '

        if 'image__save_gl_surface_test_.py' not in os.listdir('.'):
            cmd += 'test/'
        else:
            cmd += ''

        cmd += "image__save_gl_surface_test_.py"

        stdin, ret = os.popen4(cmd)
        stdin.close()        
        
        try: ret.seek(0)
        except IOError: pass
        
        gl_surface_save_test = ret.read().strip()
        ret.close()
        
        if "Segmentation Fault" in gl_surface_save_test:
            raise Exception('Segmentation Fault')

#        self.assert_(gl_surface_save_test.endswith('OK'))
        ok = re.match(r'\s*KO', gl_surface_save_test[-1::-1]) is not None
        self.assert_(ok)

if __name__ == '__main__': 
    unittest.main()
