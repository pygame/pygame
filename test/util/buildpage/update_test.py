import unittest
import os

import update
import config

def mock_config(*args, **kw):
    if args: kw = args[0]
    update.config = config.config_obj(kw)

def fixture(f):
    return open(os.path.join('test_fixtures', f)).read()

class UpdateBuildsTest(unittest.TestCase):
    def setUp(self):
        reload(update)

    def test_can_mock_global_config_object(self):
        self.assertRaises(Exception, update.build)

        mock_config (
            build_cmd = ['echo', 'mochachino'], 
            src_path  = '/',
            build_env = os.environ,
        )

        update.build()

    def test_parse_test_results__good_config(self):
        test_output = fixture('passing_tests')
        ret_code = 0

        result, errors = update.parse_test_results(ret_code, test_output)

        self.assert_(result is update.TESTS_PASSED)
        
    def test_parse_build_results__jibberish(self):
        build_output = 'arst'
        ret_code = 1
        result, errors = update.parse_build_results(ret_code, build_output)
    
        self.assert_(result is update.BUILD_FAILED_UNPARSEABLE)
        
if __name__ == '__main__':
    unittest.main()