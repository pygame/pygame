import unittest
from pygame.threads import FuncResult, tmap, WorkerQueue, Empty, STOP
from pygame import threads
from pygame.compat import xrange_

import time


class WorkerQueueTypeTest(unittest.TestCase):
    def test_usage_with_different_functions(self):
        def f(x):
            return x+1

        def f2(x):
            return x+2

        wq = WorkerQueue()
        fr = FuncResult(f)
        fr2 = FuncResult(f2)
        wq.do(fr, 1)
        wq.do(fr2, 1)
        wq.wait()
        wq.stop()

        self.assert_(fr.result  == 2)
        self.assertEqual(fr2.result, 3)

    def test_do(self):

        # __doc__ (as of 2008-06-28) for pygame.threads.WorkerQueue.do:

          # puts a function on a queue for running later.
          #
        return

    def test_stop(self):

        # __doc__ (as of 2008-06-28) for pygame.threads.WorkerQueue.stop:

          # Stops the WorkerQueue, waits for all of the threads to finish up.
          #

        wq = WorkerQueue()

        self.assert_(len(wq.pool) > 0)
        for t in wq.pool: self.assert_(t.isAlive())

        for i in xrange_(200): wq.do(lambda x: x+1, i)

        wq.stop()
        for t in wq.pool: self.assert_(not t.isAlive())

        self.assert_(wq.queue.get() is STOP)

    def todo_test_threadloop(self):

        # __doc__ (as of 2008-06-28) for pygame.threads.WorkerQueue.threadloop:

          # Loops until all of the tasks are finished.

        self.fail()

    def test_wait(self):

        # __doc__ (as of 2008-06-28) for pygame.threads.WorkerQueue.wait:

          # waits until all tasks are complete.

        wq = WorkerQueue()

        for i in xrange_(2000): wq.do(lambda x: x+1, i)
        wq.wait()

        self.assertRaises(Empty, wq.queue.get_nowait)

        wq.stop()

class ThreadsModuleTest(unittest.TestCase):
    def todo_test_benchmark_workers(self):
        "tags:long_running"

        # __doc__ (as of 2008-06-28) for pygame.threads.benchmark_workers:

          # does a little test to see if workers are at all faster.
          # Returns the number of workers which works best.
          # Takes a little bit of time to run, so you should only really call
          #   it once.
          # You can pass in benchmark data, and functions if you want.
          # a_bench_func - f(data)
          # the_data - data to work on.

        self.fail()

    def test_init(self):

        # __doc__ (as of 2008-06-28) for pygame.threads.init:

          # Does a little test to see if threading is worth it.
          #   Sets up a global worker queue if it's worth it.
          #
          # Calling init() is not required, but is generally better to do.

        threads.init(8)

        self.assert_(isinstance(threads._wq, WorkerQueue))

        threads.quit()

    def test_quit(self):

        # __doc__ (as of 2008-06-28) for pygame.threads.quit:

          # cleans up everything.
          #

        threads.init(8)

        threads.quit()

        self.assert_(threads._wq is None)

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

        func, data = lambda x:x+1, xrange_(100)

        tmapped = list(tmap(func, data))
        mapped = list(map(func, data))

        self.assertEqual(tmapped, mapped)

    def test_tmap__None_func_and_multiple_sequences(self):
        return     #TODO

        """ Using a None as func and multiple seqences """

        res =  tmap(None, [1,2,3,4])

        res2 = tmap(None, [1,2,3,4], [22, 33, 44, 55])

        res3 = tmap(None, [1,2,3,4], [22, 33, 44, 55, 66])

        res4 = tmap(None, [1,2,3,4,5], [22, 33, 44, 55])

        self.assertEqual([1, 2, 3, 4], res)
        self.assertEqual([(1, 22), (2, 33), (3, 44), (4, 55)], res2)
        self.assertEqual([(1, 22), (2, 33), (3, 44), (4, 55), (None, 66)], res3)
        self.assertEqual([(1, 22), (2, 33), (3, 44), (4, 55), (5,None)], res4)

    def test_tmap__wait(self):
        r = range(1000)
        wq, results = tmap(lambda x:x, r, num_workers = 5, wait=False)
        wq.wait()
        r2 = map(lambda x:x.result, results)
        self.assertEqual(list(r), list(r2))

    def test_FuncResult(self):
        # as of 2008-06-28
        # FuncResult(f, callback = None, errback = None)

        # Used for wrapping up a function call so that the results are stored
        #      inside the instances result attribute.


        #     f - is the function we that we call
        #         callback(result) - this is called when the function(f) returns
        #         errback(exception) - this is called when the function(f) raises
        #                                an exception.

        # Results are stored in result attribute
        fr = FuncResult(lambda x:x+1)
        fr(2)
        self.assertEqual(fr.result, 3)

        # Exceptions are store in exception attribute
        self.assert_(fr.exception is None,  "when no exception raised")

        exception = ValueError('rast')
        def x(sdf):
            raise exception
        fr = FuncResult(x)
        fr(None)
        self.assert_(fr.exception is exception)

################################################################################

if __name__ == '__main__':
    unittest.main()
