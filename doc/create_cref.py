from xml.dom.minidom import parse
import os, sys
try:
    import cStringIO as stringio
except ImportError:
    import io as stringio

def prepare_text (text):
    newtext = ""
    lines = text.split ("\n")
    for l in lines:
        newtext += l.strip () + "\\n"
    return newtext.strip("\\n")

def create_cref (dom, buf):
    module = dom.getElementsByTagName ("module")[0]
    modname = module.getAttribute ("name")
    tmp = modname.split (".")[-1]
    headname = tmp.upper()
    docprefix = "DOC_%s_" % headname
    desc = module.getElementsByTagName ("desc")[0].firstChild.nodeValue
    desc = prepare_text (desc)
    
    buf.write ("#ifndef _PYGAME2_DOC%s_H_\n" % headname)
    buf.write ("#define _PYGAME2_DOC%s_H_\n\n" % headname)
    buf.write ("#define DOC_%s \"%s\"\n" % (headname, desc))
    
    create_func_refs (module, docprefix, buf)
    create_class_refs (module, docprefix, buf)
    
    buf.write ("\n#endif\n")

def create_func_refs (module, docprefix, buf):
    funcs = module.getElementsByTagName ("func")
    for func in funcs:
        name = func.getAttribute ("name").upper ()
        desc = func.getElementsByTagName ("desc")[0].firstChild.nodeValue
        desc = prepare_text (desc)
        buf.write ("#define %s \"%s\"\n" % (docprefix + name, desc))
    
def create_class_refs (module, docprefix, buf):
    classes = module.getElementsByTagName ("class")
    for cls in classes:
        name = cls.getAttribute ("name").upper ()
        desc = cls.getElementsByTagName ("desc")[0].firstChild.nodeValue
        desc = prepare_text (desc)
        buf.write ("#define %s \"%s\"\n" % (docprefix + name, desc))

        attrs = cls.getElementsByTagName ("attr")
        attrprefix = docprefix + name + "_"
        for attr in attrs:
            create_attr_ref (attr, attrprefix, buf)

        methods = cls.getElementsByTagName ("method")
        methodprefix = attrprefix
        for method in methods:
            create_method_ref (method, methodprefix, buf)

def create_attr_ref (attr, prefix, buf):
    name = attr.getAttribute ("name").upper ()
    desc = attr.firstChild.nodeValue
    desc = prepare_text (desc)
    buf.write ("#define %s \"%s\"\n" % (prefix + name, desc))
    
def create_method_ref (method, prefix, buf):
    name = method.getAttribute ("name").upper ()
    desc = method.getElementsByTagName ("desc")[0].firstChild.nodeValue
    desc = prepare_text (desc)
    buf.write ("#define %s \"%s\"\n" % (prefix + name, desc))

def create_c_header (infile, outfile):
    try:
        dom = parse (infile)
        buf = stringio.StringIO ()
        create_cref (dom, buf)
        f = open (outfile, "w")
        f.write (buf.getvalue ())
        f.flush ()
        f.close ()
    except Exception:
        print (sys.exc_info()[1])
    
    
if __name__ == "__main__":
    if len (sys.argv) < 2:
        print ("usage: %s file.xml" % sys.argv[0])
    dom = parse (sys.argv[1])
    buf = stringio.StringIO ()
    create_cref (dom, buf)
    print (buf.getvalue ())
