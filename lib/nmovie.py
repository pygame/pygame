"""Main newmovie module, imports first from _movie.so, then _vlcbackend if it finds the vlc executable."""

try:
    from pygame._movie import Movie
except ImportError:
    #try to transparently load the _vlcbackend.py Movie object.
    import os, os.path, sys
    path=os.path
    if('win' in sys.platform):
        if(os.path.exists(path.join(path.join(path.join('C:', 'Program Files'), 'VideoLan'), 'VLC'))):
            try:
                from pygame._vlcbackend import Movie, MovieWatcher
            except ImportError:
                #you're really hooped now...
                print("Unable to find a working movie backend. Loading the dummy movie class...")
                from pygame._dummybackend import Movie
        else:
            print("Unable to find a working movie backend. Loading the dummy movie class...")
            from pygame._dummybackend import Movie
    else:
        #POSIX
        if(os.path.exists(path.join(path.join(path.join(os.sep, 'usr'), 'bin'), 'vlc'))):
            try:
                from pygame._vlcbackend import Movie, MovieWatcher
            except ImportError:
                #oh man, I didn't mean for this to happen so badly...
                print("Unable to find a working movie backend. Loading the dummy movie class...")
                from pygame._dummybackend import Movie
        else:
            print("Unable to find a working movie backend. Loading the dummy movie class...")
            from pygame._dummybackend import Movie
            
