# module lazyimp.keylock

"""Per-key reentrant thread locks"""

import threading

class Lock(object):
    """A per-key reentrant thread lock

    KeyLock(Key) => acquired lock

    KeyLock provides locking by key. If others threads hold a non-freed Lock
    instance for the given key then the constructor blocks until the other
    threads have freed their instances.

    """

    cache = {}
    cache_lock = threading.Lock()

    class Error(Exception):
        pass

    def __new__(cls, key):
        """Return a new acquired Lock instance"""
        cls.cache_lock.acquire()
        try:
            try:
                lock, count = cls.cache[key]
            except KeyError:
                lock = threading.RLock()
                count = 0
            cls.cache[key] = lock, count + 1
        finally:
            cls.cache_lock.release()
        obj = object.__new__(cls)
        obj.key = key
        lock.acquire()
        return obj

    def free(self):
        """Release the lock"""
        self.cache_lock.acquire()
        try:
            try:
                key = self.key
            except AttributeError:
                raise self.Error("Lock already freed")
            lock, count = self.cache.pop(key)
            lock.release()
            if count > 1:
                self.cache[key] = lock, count - 1
            del self.key
        finally:
            self.cache_lock.release()

    def __del__(self):
        """Release the lock
        
        Yes, this method is here but should not be relied upon.
        Call free explicitly.

        """
        try:
            self.free()
        except self.Error:
            pass
