"""
For finding out what is not documented.

python compare_docs.py pygame.sprite sprite.doc



An example showing how to compare what the .doc docs have in them
 with what the module has in them.

For example, to see what the sprite.doc has documented compared to what the
  sprite module actually has in it.

To be move to 
"""

import sys, os
#import pygame.sprite

#the_module = pygame.sprite
#the_doc = "sprite.doc"

try:
#if 1:
    the_module = __import__(sys.argv[-2])
    last_part = sys.argv[-2].split(".")[-1]
    print last_part
    the_module = getattr(the_module, last_part)

    the_doc = sys.argv[-1]
except:
    print "useage:\n  python compare_docs.py pygame.sprite sprite.doc"
    raise



# put makeref.py in the path.
sys.path.append(os.path.join("..", ".."))
from makeref import Doc


f = None
f2 = None
f = os.path.join("..", "..", "lib", the_doc)
if not os.path.exists(f):
    f2 = os.path.join("..", "..", "src", the_doc)
    if not os.path.exists(f2):
        raise "paths do not exist :%s:    :%s:" % (f,f2)
    else:
        f = f2

    

d = Doc('', open(f, "U"))

print d
for x in dir(d):
    print ":%s:" % x, getattr(d, x)
#def obj_doc_file():

print "\n\n"
print "Stuff in docs - the_module.  In the_module, not in docs:"
print set(dir(the_module)) - set([x.name for x in d.kids])
print "-" * 20


print "\n\n"
print "stuff in docs - the_module.   In docs, not in the_module."
print set([x.name for x in d.kids]) - set(dir(the_module)) 



print "\n\n"
print "everything in the module."
print set(dir(the_module)) 



