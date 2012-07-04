import subprocess, os, time, socket, select
import threading
import re

import sys
if('win' in sys.platform):
    player="vlc.exe"
else:
    player= "vlc"
remote= "-I rc"
port = 10000
hostname = socket.getaddrinfo('localhost', 10000)[0][4][0]
extra = "--rc-host %s:%d"

class Communicator:
    def __init__(self, player, remote, extra, port, hostname):
        self.socket = socket.socket()
        while 1:
            try:
                self.socket.connect((hostname, port))
                break
            except socket.error:
                port+=1
        self.commands =  ' '.join([player, remote, extra%(hostname, port)])
        self.patterns = {
            'size'  : re.compile("Resolution: \d{1,4}x\d{1,4}"), 
            'width' : re.compile("Resolution: \d{1,4}(?=\d{1,4})"), 
            'height': re.compile("Resolution: (?<=\dx)\d{1,4}|(?<=\d\dx)\d{1,4}|(?<=\d\d\dx)\d{1,4}|(?<=\d\d\d\dx)\d{1,4}"),
                        }
    def send(self, message):
        self.socket.send(message)

    def add(self, filename):
        self.send("add %s\n" % filename)
        
    def enqueue(self, filename):
        self.send("enqueue %s\n" % filename)
        
    def pause(self):
        self.send("pause\n")
        
    def stop(self):
        self.send("stop\n")
        
    def logout(self):
        self.send("logout\n")

    def info(self):
        self.send("info\n")
        d=[]
        read =[0]
        while(len(read)>0):
            read, write, exce = select.select([self.socket], [], [], 0.10)
            if(len(read)>0):
                d.append(self.socket.recv(1))
        d=''.join(d)
        return d
        
    def _get_time(self):
        self.send("get_time\n")
        d=[]
        read =[0]
        while(len(read)>0):
            read, write, exce = select.select([self.socket], [], [], 0.10)
            if(len(read)>0):
                d.append(self.socket.recv(1))
        d=''.join(d)
        d=int(d)#transforms into an int
        return d
    
    def _get_height(self):
        d=self.info()
        p = self.patterns['height']
        m =p.search(d)
        if not m:
            return 0
        return int(m.group())

    def _get_width(self):
        d=self.info()
        p= self.patterns['width']
        m=p.search(d)
        if not m:
            return 0
        return int(m.group())

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
        self.comm = Communicator(player, remote, extra, port, hostname)
        self.width = 0
        self.height =0
        self.finished =0
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
        comm = self.comm.commands
        self.process=subprocess.Popen(comm, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.playing = not self.playing
        self.width = self.comm._get_width()
        self.height = self.comm._get_height()
        if(self.eternity):
            #well, we need to watch vlc, see when it finishes playing, then play the video again, over and over
            self.comm.add(self.filename)
            self.t=MovieWatcher(self.comm, 0.1 , self.eternity, self.filename)
            return
        #otherwise, we add loops+1 copies of the video to the playlist
        self.comm.add(self.filename)
        for i in range(1, self.loops+1):
            self.comm.enqueue(self.filename)
        
        
    def pause(self):
        if self.process:
            #send value to pause playback
            self.paused = not self.paused
            self.comm.pause()
    
    def stop(self):
        if self.process:
            #we kill the process...
            self.paused = not self.paused
            self.playing= not self.playing
            self.comm.stop()
            self.commd.logout()
            self.process.terminate()
            self.process=None
            self.finished  = 1
            
    def __repr__(self):
        if self.process:
            return "(%s: %ds)" % (self.filename, self._get_time()) #add on timestamp
        else:
            return "(%s: 0.0s)" % self.filename
    
    def _get_time(self):
        if self.process:
            return self.comm._get_time()
        
    

class MovieWatcher(threading.Thread):
    def __init__(self, comm, time, eternity, filename):
        threading.Thread.__init__(self)
        self.comm=comm
        self.time = time
        self.eternity = eternity
        self.filename = filename
    def run():
        while(1):        
            time.sleep(self.time)
            read, write, exce = select.select([self.comm.socket], [], [], 0.1)
            d=[]
            while(len(read)>0):
                d.append(self.comm.socket.recv(1))
                read, write, exce = select.select([self.comm.socket], [], [], 0.1)
            s = ''.join(d)
            if("status change: ( stop state: 0 )" in s): 
                if("nothing to play" in s):
                    self.comm.add(self.filename)
                    
