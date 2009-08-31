#ifndef _MOVIE_DOC_H_
#define _MOVIE_DOC_H_

#define DOC_GMOVIE "pygame backend module that wraps the ffmpeg library to play video files"

#define DOC_GMOVIEMOVIE "pygame._movie.Movie(filename, surface=None): returns Movie or None\nIf the optional surface argument is a surface, then the movie will output to that surface instead of using overlays."

#define DOC_GMOVIEMOVIEPLAY "pygame._movie.Movie.play(loops=0): return None\nplays the video file loops+1 times."

#define DOC_GMOVIEMOVIESTOP "pygame._movie.Movie.stop(): return None\nstops the video file and returns it to timestamp o."

#define DOC_GMOVIEMOVIEPAUSE "pygame._movie.Movie.pause(): return None\npauses video file at that very moment or unpauses the video file."

#define DOC_GMOVIEMOVIEREWIND "pygame._movie.Movie.rewind(): return None\nsame as stop()"

#define DOC_GMOVIEMOVIERESIZE "pygame._movie.Movie.resize(width, height): return None\nresizes the video screen. If a surface has been provided, then resize will not work, to prevent image corruption issues.\nYou would need to provide a new surface to change the size."

#define DOC_GMOVIEMOVIEPAUSED "pygame._movie.Movie.paused: return bool\nchecks if the movie file has been paused"

#define DOC_GMOVIEMOVIEPLAYING "pygame._movie.Movie.playing: return bool\nchecks if the movie file is playing. True even when paused, but false when stop has been called."

#define DOC_GMOVIEMOVIEWIDTH   "pygame._movie.Movie.width: Gets or sets the width\nGet or set the width of the screen for the video playback"

#define DOC_GMOVIEMOVIEHEIGHT  "pygame._movie.Movie.height: Gets or sets the height\nGet or set the height of the screen for the video playback"

#define DOC_GMOVIEMOVIESURFACE "pygame._movie.Movie.surface: Gets or sets the surface to which the video is displayed on."

#define DOC_GMOVIEMOVIEFINISHED "pygame._movie.Movie.finished: Indicates when the video is played.\n If using multiple plays, this is not a reliable member to use, as when a video ends, regardless of if there are further plays, the finished member is triggered."

#define DOC_GMOVIEMOVIEYTOP  "pygame._movie.Movie.ytop: Gets or sets the ytop of the display rect\nThis sets the distance between the image and the top of the window. Increase it to move the image down, or decrease it to move the image up."

#define DOC_GMOVIEMOVIEXLEFT "pygame._movie.Movie.xleft: Gets or sets the xleft of the display rect\nThis sets the distance between the image and the left of the window. Increase it to move the image right, or decrease it to move the image left."

#define DOC_GMOVIEMOVIEEASY_SEEK "pygame._movie.Movie.easy_seek(second, minute, hour, reverse): return None\nThis is a non-relative seek, instead seeking to the h:mm:ss timestamp on the video as given. All arguments are needed."

#define DOC_GMOVIEMOVIESHIFT "pygame._movie.Movie.shift(ytop, xleft): return None\nShift the video image up, left, right, or down. Default values are 0,0."


#endif /*_MOVIE_DOC_H_*/
