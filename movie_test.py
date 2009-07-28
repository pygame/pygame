import time 
import pygame
pygame.init() #or we could just call pygame.display.init() as thats all we need
pygame.mixer.quit() #This needs to be done, as the default sound system of the 
                    # ffmpeg-wrapper uses SDL_mixer, and this would cause major
                    # conflicts with the mixer module.
import pygame._movie as movie

print "Please give an (absolute)filename of a movie file you'd like to play: ",
#filename = raw_input()
filename="/home/tyler/dhs1.avi"
#initialization. It could also have a surface as a second argument, and every 
# frame will be blitted to that surface. It is the programmer's responsibility
# to be on time for rendering that surface.
# Without a surface argument, the ffmpeg-wrapper uses the sdl_overlay library. 
#screen=pygame.display.set_mode((640, 368))

m = movie.Movie(filename)
print m.paused  #always False, unless .pause has been called
print m.playing #False until play has been called. Will return to false when
print m.finished# .stop() has been called.
                
print m.width   #default size values of the video file
print m.height  # They can be modified on the fly, as will be demonstrated.

print m         #calls __repr__, which will show the filename, and the current 
                # timestamp. 
#print "repeated looping plays.."#
#m.play(10)
#time.sleep(10*30)


print "Playing infinitely"

m.play(-1)       #We're going to use infinite play, so we can demonstrate all 
                # the features.
time.sleep(2)  #sleep for ten seconds to let one see the video play, and hear 
                # the audio
##print "Paused:",m.paused
##print "Playing:",m.playing
##print "Movie:",m
##print "Y Top:",m.ytop
##print "X Left:",m.xleft
time.sleep(30)
print "Testing seek..."
m.easy_seek(second=10, minute=5, reverse=0)
time.sleep(5)

print "Altering xleft and ytop..."
m.xleft += 10
m.ytop  +=10
time.sleep(10)
m.xleft -= 10
m.ytop -=10
#Now we're going to play with the size of the window, affecting the video on 
#the fly. resize(width, height) is the main function, changes them both at
# the same time.
print "Resizing..."
m.resize(m.width/2, m.height*2)
print "sleeping..."
time.sleep(10) #another ten second nap.
print "Resizing again..."
m.width = m.width*4
print "sleeping again" 
time.sleep(10)
print "Back to normal!"
m.width=m.width/2
m.height = m.height/2
print "and again, sleeping..."
#back to our original size
time.sleep(10)


#Here we demonstrate the use of pause. You pause, then call pause again to play
##print "Pausing..."
##m.pause()
##print "done pausing..."
##print m.paused
##print m.playing
##time.sleep(10)
##print "Unpausing..."
##m.pause()
##print m.paused
##print m.playing
##time.sleep(10)
###Here is the stop function. Right now, rewind is the exact same as stop.
##print "Stopping..., sleeping for 3 seconds"
##m.stop()
##time.sleep(3)
###And now we restart playing.
##print "Playing again..." 
##m.play(-1)
##print "done restart play..."
##time.sleep(10)
print "Surface time..."
screen = pygame.display.set_mode((640, 348))
#This will move the movie player from overlay mode to blitting to the surface 
# we've given it. This means it is our responsibility to update the display on 
# time.

m=movie.Movie(filename, screen)
counter = 0
actions = {1: lambda x: x.paused, 6: lambda x:x.pause(), 11: lambda x:x.pause(), 15: lambda x: x.resize(int(x.width*0.9), int(x.height*0.9)), 2000:lambda x: x.stop(), 3000: lambda x: x.play(-1)}
m.play(0)
prev_time = time.time()
#m.resize(m.width*2, m.height*2)
#m.surface = screen
print "About to do surface gymnastics..."
while not m.finished:
    time.sleep(0.1)
    pygame.display.update()
time.sleep(1000)
while(1):
    new_time=time.time()
    diff = int(new_time-prev_time)
    if(diff>=1):
        counter+=1
        print counter
        prev_time=new_time
    #print "testing counter"
    if counter==3100:
        #print "breaking"
        break
    #print "has_key"
    if actions.has_key(counter):
        print "Performing action at counter value: %d" % counter
        actions[counter](m)
        counter +=1
    #print "updating"
    time.sleep(0.1) #we need to let go of the gil occassionally...
    if(not screen.get_locked()):
        try:
            pygame.display.update() #we can do this because we're blitting each frame of the movie to the main screen we instantiated.
        except pygame.error:
            break
        
print "Ending trial one..."
m.stop()
del m
#the end

m=movie.Movie(filename, screen)
prev_time = time.time()
print "About to do surface gymnastics..."
counter = 0
while(1):
    new_time=time.time()
    diff = int(new_time-prev_time)
    if(diff>=1):
        counter+=1
        print counter
        prev_time=new_time
    #print "testing counter"
    if counter==30:
        print "breaking"
        break
    #print "has_key"
    if actions.has_key(counter):
        print "Performing action at counter value: %d" % counter
        actions[counter](m)
        counter +=1
    #print "updating"
    time.sleep(0.1) #we need to let go of the gil occassionally...
    if(not screen.get_locked()):
        try:
            pygame.display.update() #we can do this because we're blitting each frame of the movie to the main screen we instantiated.
        except pygame.error:
            break

del m
print "the end"
