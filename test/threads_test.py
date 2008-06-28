#################################### IMPORTS ###################################

import test_utils, unittest
from test_utils import test_not_implemented

################################################################################

class WorkerQueueTypeTest(unittest.TestCase):
    def test_do(self):

        # __doc__ (as of 2008-06-28) for pygame.threads.WorkerQueue.do:

          # puts a function on a queue for running later.
          #         

        self.assert_(test_not_implemented()) 

    def test_stop(self):

        # __doc__ (as of 2008-06-28) for pygame.threads.WorkerQueue.stop:

          # Stops the WorkerQueue, waits for all of the threads to finish up.
          #         

        self.assert_(test_not_implemented()) 

    def test_threadloop(self):

        # __doc__ (as of 2008-06-28) for pygame.threads.WorkerQueue.threadloop:

          # Loops until all of the tasks are finished.
          #         

        self.assert_(test_not_implemented()) 

    def test_wait(self):

        # __doc__ (as of 2008-06-28) for pygame.threads.WorkerQueue.wait:

          # waits until all tasks are complete.
          #         

        self.assert_(test_not_implemented()) 

class ThreadsModuleTest(unittest.TestCase):
    def test_benchmark_workers(self):

        # __doc__ (as of 2008-06-28) for pygame.threads.benchmark_workers:

          # does a little test to see if workers are at all faster.
          # Returns the number of workers which works best.
          # Takes a little bit of time to run, so you should only really call
          #   it once.
          # You can pass in benchmark data, and functions if you want.
          # a_bench_func - f(data)
          # the_data - data to work on.

        self.assert_(test_not_implemented()) 

    def test_init(self):

        # __doc__ (as of 2008-06-28) for pygame.threads.init:

          # Does a little test to see if threading is worth it.
          #   Sets up a global worker queue if it's worth it.
          # 
          # Calling init() is not required, but is generally better to do.

        self.assert_(test_not_implemented()) 

    def test_quit(self):

        # __doc__ (as of 2008-06-28) for pygame.threads.quit:

          # cleans up everything.
          #     

        self.assert_(test_not_implemented()) 

    def test_tmap(self):

        # __doc__ (as of 2008-06-28) for pygame.threads.tmap:

          # like map, but uses a thread pool to execute.
          # num_workers - the number of worker threads that will be used.  If pool
          #                 is passed in, then the num_workers arg is ignored.
          # worker_queue - you can optionally pass in an existing WorkerQueue.
          # wait - True means that the results are returned when everything is finished.
          #        False means that we return the [worker_queue, results] right away instead. 
          #        results, is returned as a list of FuncResult instances.
          # stop_on_error - 

        self.assert_(test_not_implemented()) 

################################################################################

if __name__ == '__main__':
    test_utils.get_fail_incomplete_tests_option()
    unittest.main()