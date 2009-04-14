from xml.dom.minidom import parse
import os, glob, sys

RST_HEADER = """"""

RST_FOOTER = """"""

class DocClass (object):
    def __init__ (self, name, constructor, description):
        self.name = name
        self.constructor = constructor
        self.description = description
        self.attributes = []
        self.methods = []
    
    def __repr__ (self):
        return "<DocClass '%s'>" % self.name

class DocAttribute (object):
    def __init__ (self, name, description):
        self.name = name
        self.description = description

    def __repr__ (self):
        return "<DocAttribute '%s'>" % self.name

class DocMethod (object):
    def __init__ (self, cls, name, call, description):
        self.name = name
        self.calls = []
        for line in call.split ("\n"):
            line = line.strip ()
            if len (line) > 0:
                self.calls.append (line)
        self.cls = cls
        self.description = description

    def __repr__ (self):
        if self.cls:
            return "<DocMethod '%s.%s'>" % (self.cls, self.name)
        return "<DocMethod '%s'>" % (self.name)

class Doc(object):
    def __init__ (self, filename):
        self.filename = filename
        self.modulename = None
        self.shortdesc = "TODO"
        self.description = "TODO"
        self.classes = []
        self.functions = []
    
    def parse_content (self):
        dom = parse (self.filename)
        module = self.get_module_docs (dom)
        self.get_module_funcs (module)
        self.get_class_refs (module)
    
    def get_module_docs (self, dom):
        module = dom.getElementsByTagName ("module")[0]
        self.modulename = module.getAttribute ("name")
        node = module.getElementsByTagName ("short")[0]
        if node.firstChild:
            self.shortdesc = node.firstChild.nodeValue
        node = module.getElementsByTagName ("desc")[0]
        if node.firstChild:
            self.description = node.firstChild.nodeValue
        return module
    
    def get_module_funcs (self, module):
        functions = module.getElementsByTagName ("func")
        for func in functions:
            name = func.getAttribute ("name")
            if len (name) == 0:
                name = "TODO"
            node = func.getElementsByTagName ("call")[0]
            if node.firstChild:
                call = node.firstChild.nodeValue
            else:
                call = "TODO"
            node = func.getElementsByTagName ("desc")[0]
            if node.firstChild:
                desc = node.firstChild.nodeValue
            else:
                desc = "TODO"
            self.functions.append (DocMethod (None, name, call, desc))

    def get_class_refs (self, module):
        classes = module.getElementsByTagName ("class")
        for cls in classes:
            name = cls.getAttribute ("name")
            if len (name) == 0:
                name = "TODO"
            node = cls.getElementsByTagName ("constructor")[0]
            if node.firstChild:
                const = node.firstChild.nodeValue
            else:
                const = "TODO"
            node = cls.getElementsByTagName ("desc")[0]
            if node.firstChild:
                desc = node.firstChild.nodeValue
            else:
                desc = "TODO"
            clsdoc = DocClass (name, const, desc)

            attrs = cls.getElementsByTagName ("attr")
            for attr in attrs:
                self.create_attr_ref (clsdoc, attr)

            methods = cls.getElementsByTagName ("method")
            for method in methods:
                self.create_method_ref (clsdoc, method)
            
            self.classes.append (clsdoc)

    def create_attr_ref (self, doccls, attr):
        name = attr.getAttribute ("name")
        if len (name) == 0:
            name = "TODO"
        if attr.firstChild:
            desc = attr.firstChild.nodeValue
        else:
            desc = "TODO"
        doccls.attributes.append (DocAttribute (name, desc))

    def create_method_ref (self, doccls, method):
        name = method.getAttribute ("name")
        if len (name) == 0:
            name = "TODO"
        node = method.getElementsByTagName ("call")[0]
        if node.firstChild:
            call = node.firstChild.nodeValue
        else:
            call = "TODO"
        node = method.getElementsByTagName ("desc")[0]
        if node.firstChild:
            desc = node.firstChild.nodeValue
        else:
            desc = "TODO"
        doccls.methods.append (DocMethod (doccls, name, call, desc))

    def create_desc_rst (self, desc, offset=0):
        written = 0
        data = ""
        cindent = indent = 0
        insrcblock = False
        
        for line in desc.split ("\n"):
            line = line.rstrip ()
            if len (line) != 0:
                # Prepare indentation stuff
                cindent = len (line) - len (line.lstrip ())
            
            if cindent < indent:
                # Left the current block, reset the markers
                insrcblock = False
                indent = cindent
                
            # Check whether a source code block is about to start
            if not insrcblock:
                line = line.lstrip ()
                insrcblock = line.endswith ("::")
                if insrcblock:
                    # We are in a source code block. update the indentation
                    indent = cindent + 1
                    cindent += 1 # Set for following, empty lines.
            
            if written > 0 and line == "":
                data += "\n"
            elif line != "":
                data += "\n" + " " * offset + line
                written += 1
        if written == 0:
            return ""
        return data + "\n\n"
    
    def create_func_rst (self, func):
        data = ".. function:: %s\n" % func.calls[0]
        for call in func.calls[1:]:
            data += "              %s\n" % call
        data += "%s\n" % self.create_desc_rst (func.description, 2)
        return data
    
    def create_rst (self):
        fname = os.path.join ("ref", "%s.rst")
        fp = open (fname % self.modulename.replace (".", "_"), "w")
        fp.write (RST_HEADER)
        
        fp.write (":mod:`%s` -- %s\n" % (self.modulename, self.shortdesc))
        fp.write ("%s\n" %
                  ("=" * (11 + len (self.modulename) + len (self.shortdesc))))
        fp.write ("%s" % self.create_desc_rst (self.description))
        fp.write (".. module:: %s\n" % (self.modulename))
        fp.write ("   :synopsis: %s\n\n" % (self.shortdesc))
        
        if len (self.functions) > 0:
            fp.write ("Module functions\n")
            fp.write ("----------------\n")
            for func in self.functions:
                fp.write (self.create_func_rst (func))

        if len (self.classes) > 0:
            for cls in self.classes:
                fp.write (cls.name + "\n")
                fp.write ("%s\n" % ("-" * len (cls.name)))
                fp.write (".. class:: %s\n" % cls.constructor)
                fp.write (self.create_desc_rst (cls.description, 2))
                if len (cls.attributes) > 0:
                        fp.write ("Attributes\n")
                        fp.write ("^^^^^^^^^^\n")
                        for attr in cls.attributes:
                            fp.write (".. attribute:: %s.%s\n"
                                      % (cls.name, attr.name))
                            fp.write (self.create_desc_rst (attr.description, 2))
                if len (cls.methods) > 0:
                    fp.write ("Methods\n")
                    fp.write ("^^^^^^^\n")
                    for method in cls.methods:
                        fp.write (".. method:: %s.%s\n" %
                                  (cls.name, method.calls[0]))
                        for call in method.calls[1:]:
                            fp.write ("            %s.%s\n" % (cls.name, call))
                        fp.write (self.create_desc_rst (method.description, 2))
        fp.write ("\n")
        fp.write (RST_FOOTER)
        fp.close ()
        
def create_rst (docs):
    if not os.path.exists ("ref"):
        os.mkdir ("ref")
    for doc in docs:
        print ("Now writing RST for %s...." % doc.modulename)
        doc.create_rst ()

def get_doc_files ():
    docs = []
    files = glob.glob (os.path.join ("src", "*.xml"))
    for fname in files:
        docs.append (Doc (fname))
    return docs

if __name__ == "__main__":
    docs = get_doc_files ()
    for doc in docs:
        print ("Parsing file %s..." % doc.filename)
        doc.parse_content ()
    create_rst (docs)
