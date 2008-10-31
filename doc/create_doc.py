from inspect import *
import os, sys
try:
    import cStringIO as stringio
except ImportError:
    import io as stringio

if sys.version_info < (2,5,0):
    ismemberdescriptor = isdatadescriptor
    isgetsetdescriptor = isdatadescriptor

def do_import (name):
    components = name.split ('.')
    classed = False
    try:
        mod = __import__ (name)
    except ImportError:
        # Perhaps a class name?
        mod = __import__ (".".join (components[:-1]))
        classed = True
    if len (components) > 1:
        if classed:
            parts = components[1:-1]
        else:
            parts = components[1:]
        for comp in parts:
            mod = getattr (mod, comp)
    return mod

def write_dtd (buf):
    buf.write ("<!DOCTYPE module [\n")
    buf.write ("<!ELEMENT module (desc, (func|class)*)>\n")
    buf.write ("<!ATTLIST module name CDATA #REQUIRED>\n")
    buf.write ("<!ELEMENT desc (#PCDATA)>\n")
    buf.write ("<!ELEMENT func (desc)>\n")
    buf.write ("<!ATTLIST func name CDATA #REQUIRED>\n")
    buf.write ("<!ELEMENT class (constructor, desc, (attr|method)*)>\n")
    buf.write ("<!ATTLIST class name CDATA #REQUIRED>\n")
    buf.write ("<!ELEMENT constructor (#PCDATA)>\n")
    buf.write ("<!ELEMENT attr (#PCDATA)>\n")
    buf.write ("<!ATTLIST attr name CDATA #REQUIRED>\n")
    buf.write ("<!ELEMENT method (desc)>\n")
    buf.write ("<!ATTLIST method name CDATA #REQUIRED>\n")
    buf.write ("]>\n\n")

def document_class (cls, buf):
    buf.write ("  <class name=\"%s\">\n" % cls.__name__)
    buf.write ("    <constructor>TODO</constructor>\n")
    buf.write ("    <desc>%s</desc>\n" % cls.__doc__)
    parts = dir (cls)
    for what in parts:
        # Skip private parts.
        if what.startswith ("_"):
            continue
        try:
            obj = cls.__dict__[what]
        except:
            continue # Skip invalid ones
        if isgetsetdescriptor (obj) or ismemberdescriptor (obj):
            document_attr (obj, buf, 4)
        elif ismethod (obj) or ismethoddescriptor (obj):
            document_method (obj, buf, 4)
        else:
            pass
    buf.write ("  </class>\n\n")

def document_attr (attr, buf, indent):
    buf.write \
        (" " * indent + "<attr name=\"%s\">%s</attr>\n" %
         (attr.__name__, attr.__doc__))

def document_func (func, buf, indent):
    iindent = indent + 2
    buf.write (" " * indent + "<func name=\"%s\">\n" % func.__name__)
    buf.write (" " * iindent + "<desc>%s</desc>\n" % func.__doc__)
    buf.write (" " * indent + "</func>\n")

def document_method (method, buf, indent):
    iindent = indent + 2
    buf.write (" " * indent + "<method name=\"%s\">\n" % method.__name__)
    buf.write (" " * iindent + "<desc>%s</desc>\n" % method.__doc__)
    buf.write (" " * indent + "</method>\n")

def document_module (module, buf):

    # Module header.
    buf.write ("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n")
    write_dtd (buf)
    
    buf.write ("<module name=\"%s\">\n" % module.__name__)
    buf.write ("  <desc>%s</desc>\n\n" % module.__doc__)

    parts = dir (module)
    
    for what in parts:
        if what.startswith ("_"):
            continue # Skip private ones.

        obj = module.__dict__[what]
        if isclass (obj):
            document_class (obj, buf)
        elif isfunction (obj) or isbuiltin (obj):
            document_func (obj, buf, 2)
        elif ismethod (obj) or ismethoddescriptor (obj):
            document_method (obj, buf, 2)

    buf.write ("</module>\n")
    
if __name__ == "__main__":
    if len (sys.argv) < 2:
        print ("usage: %s module" % sys.argv[0])
    mod = do_import (sys.argv[1])
    buf = stringio.StringIO ()
    document_module (mod, buf)
    print (buf.getvalue ())
