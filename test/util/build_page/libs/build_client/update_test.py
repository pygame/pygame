import unittest
import os

import update
import config
import helpers

def mock_config(*args, **kw):
    if args: kw = args[0]
    update.config = config.config_obj(kw)

def fixture(f):
    return open(os.path.join('test_fixtures', f)).read()

class HelpersTest(unittest.TestCase):
    def test_create_zip(self):
        helpers.create_zip (
            'test.zip',

           * ('update_test.py',
            'update.py' ),
            
            **{ 'run_tests__output.txt'   :    'test',
                'setup_py__output.txt'    :    'test',
                'build_config.txt'        :    'test' }
        )

        helpers.create_zip ('test.zip', 'update_test.py', 'update.py' )

        helpers.create_zip (
            'test.zip', 
            
            'update_test.py', 'update.py', 
            
            some_var = 'YEAH'
        )

class UpdateBuildsTest(unittest.TestCase):
    def setUp(self):
        reload(update)

    def test_can_mock_global_config_object(self):
        self.assertRaises(Exception, update.build)

        mock_config (
            build_cmd = ['echo', 'mochachino'], 
            src_path  = '.',
            build_env = os.environ,
        )

        update.build()

    def test_BUILD_SUCCESSFUL(self):
        ret_code, output = 0, fixture('BUILD_SUCCESSFUL')
        result, errors = update.parse_build_results(ret_code, output)
        self.assert_(result is update.BUILD_SUCCESSFUL)

    def test_BUILD_FAILED_UNPARSEABLE(self):
        build_output = 'arst'
        ret_code = 1
        result, errors = update.parse_build_results(ret_code, build_output)
        
        self.assert_(result is update.BUILD_FAILED_UNPARSEABLE)

    def test_TESTS_PASSED(self):
        test_output = fixture('TESTS_PASSED')
        ret_code = 0
        
        result, errors = update.parse_test_results(ret_code, test_output)
        
        self.assert_(result is update.TESTS_PASSED)

    def test_incomplete(self):
        self.fail('tests_are_incomplete')

    def test_BUILD_FAILED(self):
        pass
        # self.assert_(result is update.BUILD_FAILED)
    
    def test_BUILD_LINK_FAILED(self):
        pass
        # self.assert_(result is update.BUILD_LINK_FAILED)
    
    def test_BUILD_FAILED_EXCEPTION(self):
        pass
        # self.assert_(result is update.BUILD_FAILED_EXCEPTION)
    
    def test_BUILD_FAILED_UNKNOWN  (self):
        pass
        # self.assert_(result is update.BUILD_FAILED_UNKNOWN)
    
    def test_TESTS_FAILED(self):
        pass
        # self.assert_(result is update.TESTS_FAILED)

    def test_TESTS_INVALID(self):
        pass
        # self.assert_(result is update.TESTS_INVALID)
            
if __name__ == '__main__':
    unittest.main()