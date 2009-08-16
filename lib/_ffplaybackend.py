import subprocess, os, time
import threading

player="ffplay"


class Movie(object):
    """pygame._ffmovbackend.Movie:
        plays a video file via subprocess and a pre-packaged ffplay executable.
    """
    def __init__(self, filename, surface):
        self.filename=filename
        self._surface = surface
        self.process = None
        self.loops=0
        self.playing = False
        self.paused  = False
        self._backend = "FFPLAY"
    def getSurface(self):
        #special stuff here
        return self._surface
    
    def setSurface(self, value):
        #special stuff here, like redirecting movie output here
        self._surface = value
    def delSurface(self):
        del self._surface
    surface=property(fget=getSurface, fset=setSurface, fdel=delSurface)
    
    def play(self, loops=0):
        self.loops=loops
        if loops<=-1: self.eternity=1
        else:         self.eternity=0
        self.loops -= 1
        if not self.process:
            self._play()
        #otherwise stop playback, and start again with the new loops value.
        else:
            self.stop()
            self._play()
    def _play(self):
        self.process=subprocess.Popen([player, self.filename], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.t=MovieWatcher(self, self.process.pid)
        self.t.start()
        self.playing = not self.playing
        
    def pause(self):
        if self.process:
            #send value to pause playback
            self.paused = not self.paused
            pass
    
    def stop(self):
        if self.process:
            pass
            
    def __repr__(self):
        if self.process:
            return "(%s: )" % self.filename #add on timestamp
        else:
            return "(%s)" % self.filename
    
        
class MovieWatcher(threading.Thread):
    def __init__(self, movie, pid):
        threading.Thread.__init__(self)
        self.movie=movie
        self.pid=pid
    def run():        
        sts=os.waitpid(self.pid, 0)
        #video finished playing, so we run it again
        if(movie.loops>-1 or self.eternity):
            movie._play()
            movie.loops -= 1

            