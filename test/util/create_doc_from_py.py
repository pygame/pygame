"""

For creating .doc documentation from .py files that include documentation.


"""



import sys, os, textwrap
#import pygame.sprite

#the_module = pygame.sprite
#the_doc = "sprite.doc"

try:
#if 1:
    the_module = __import__(sys.argv[-1])
    last_part = sys.argv[-1].split(".")[-1]
    the_module = getattr(the_module, last_part)

except:
    print "useage:\n  python create_doc_from_py.py pygame.sprite"
    raise



# put makeref.py in the path.
sys.path.append(os.path.join("..", ".."))
from makeref import Doc

out = []
out += [the_module.__doc__, "<SECTION>\n\n\n"]
#out += [the_module.__doc__]

if hasattr(the_module, "__all__"):
    all_attrs = the_module.__all__
else:
    all_attrs = dir(the_module)


def get_docs(the_object, all_attrs, find_children = 1):
    out = []

    for d in all_attrs:

        if d.startswith("_"):
            continue
        a = getattr(the_object, d)

        # skip constants.
        if type(a) == type(1):
            continue
        #print a
        #print d
        if not a.__doc__:
            raise Exception(repr(a))
        parts = a.__doc__.split("\n")

        # remove some parts at the end that are empty lines.
        while parts:
            #print "parts[-1] :%s:" % parts[-1]
            if not parts[-1].strip():
                #print "del!"
                del parts[-1]
            else:
                break

        text = parts[0] + "\n" + textwrap.dedent( "\n".join(parts[1:]))


        #out += [d, text , "<END>\n\n\n"]
        out += [d, text]


        # get any children...
        #if find_children:
        if hasattr(the_object, "__theclasses__"):
            if d in the_object.__theclasses__:
                out += ["<SECTION>\n\n\n"]
                children_attrs = dir(a)
                #print children_attrs
                out += get_docs(a, children_attrs, 0)
                out[-1] = out[-1][:-3]
                out += ["<END>\n\n\n"]
            else:
                pass
                #out += ["<END>\n\n\n"]
                out += ["<END>\n\n\n"]
        else:
            out += ["<END>\n\n\n"]


    return out


out += get_docs(the_module, all_attrs)


print "\n".join(out)

