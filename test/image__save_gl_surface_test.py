import unittest, os, test_utils

class GL_ImageSave(unittest.TestCase):
    def test_image_save_works_with_opengl_surfaces(self):
        
        if 'image__save_gl_surface_test_.py' not in os.listdir('.'):
            cmd = 'test/'
        else:
            cmd = ''

        cmd += "image__save_gl_surface_test_.py"

        stdin, ret = os.popen4(cmd)
        stdin.close()        
        ret.seek(0)

        gl_surface_save_test = ret.read().strip()
        ret.close()
        
        if "Segmentation Fault" in gl_surface_save_test:
            raise Exception('Segmentation Fault')

        self.assert_(gl_surface_save_test.endswith('OK'))

if __name__ == '__main__': 
    unittest.main()