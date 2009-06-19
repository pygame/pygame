import subprocess, os, time, socket, select
import threading

import sys
if('win' in sys.platform):
    player="vlc.exe"
else:
    player= "vlc"
remote= "-I rc"
port = 10000
hostname = socket.getaddrinfo('localhost', 10000)[0][4][0]
extra = "--rc-host %s:%d" % (hostname, 10000)
commands = [player, remote, extra]
print commands

class Movie(object):
    """pygame._vlcbackend.Movie:
        plays a video file via subprocess and the available vlc executable
    """
    def __init__(self, filename, surface=None):
        self.filename=filename
        self._surface = surface
        self.process = None
        self.loops=0
        self.playing = False
        self.paused  = False
        self._backend = "VLC"
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
        print ' '.join(commands)
        self.process=subprocess.Popen(' '.join(commands), shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.playing = not self.playing
        self.socket = socket.socket()
        print (hostname, port)
        self.socket.connect((hostname, port))
        if(self.eternity):
            #well, we need to watch vlc, see when it finishes playing, then play the video again, over and over
            self.socket.send("add %s\n" % self.filename)
            self.t=MovieWatcher(movie, 1 , eternity)
            return
        #otherwise, we add loops+1 copies of the video to the playlist
        self.socket.send("add %s\n" % self.filename)
        for i in range(1, self.loops+1):
            self.socket.send("enqueue %s\n" % self.filename)
            
    def pause(self):
        if self.process:
            #send value to pause playback
            self.paused = not self.paused
            self.socket.send("pause\n")
    
    def stop(self):
        if self.process:
            #we kill the process...
            self.paused = not self.paused
            self.playing= not self.playing
            self.socket.send("stop\n")
            self.socket.send("logout\n")
            self.process.terminate()
            self.process=None
            
    def __repr__(self):
        if self.process:
            return "(%s: %d)" % (self.filename, self._get_time()) #add on timestamp
        else:
            return "(%s: 0.0s)" % self.filename
    
    def _get_time(self):
        if self.process:
            self.socket.send("get_time\n")
            d=[]
            read =[0]
            while(len(read)>0):
                read, write, exce = select.select([self.socket], [], [], 0.10)
                if(len(read)>0):
                    d.append(self.socket.recv(1))
            d=''.join(d)
            d=int(d)#transforms into an int
            return d
        
            
class MovieWatcher(threading.Thread):
    def __init__(self, movie, time, eternity):
        threading.Thread.__init__(self)
        self.movie=movie
        self.time = time
        self.eternity = eternity
    def run():
        while(1):        
            time.sleep(self.time)
            read, write, exce = select.select([self.movie.socket], [], [], 0.1)
            d=[]
            while(len(read)>0):
                d.append(self.movie.socket.recv(1))
                read, write, exce = select.select([self.movie.socket], [], [], 0.1)
            s = ''.join(d)
            if("status change: ( stop state: 0 )" in s): 
                if("nothing to play" in s):
                    self.movie.socket.send("add %s\n" % self.movie.filename)
                    