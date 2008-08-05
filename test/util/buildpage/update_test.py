import unittest
import update


class StuffTest(unittest.TestCase):
    def setUp(self):
        reload(update)
    
    def test_something(self):
        pass
        # config = update.config_obj(
        #     dict (
                
        #     )
        # )
        
        # update.config
        
if __name__ == '__main__':
    unittest.main()