from xml.dom.minidom import parse
import os, glob, sys, shutil

RST_HEADER = """"""

RST_FOOTER = """"""

class DocClass (object):
    def __init__ (self, name, constructor, description, example):
        self.name = name
        self.constructor = constructor
        self.description = description
        self.attributes = []
        self.methods = []
        self.example = example
    
    def __repr__ (self):
        return "<DocClass '%s'>" % self.name

class DocAttribute (object):
    def __init__ (self, name, description, example):
        self.name = name
        self.description = description
        self.example = example

    def __repr__ (self):
        return "<DocAttribute '%s'>" % self.name

class DocMethod (object):
    def __init__ (self, cls, name, call, description, example):
        self.name = name
        self.calls = []
        for line in call.split ("\n"):
            line = line.strip ()
            if len (line) > 0:
                self.calls.append (line)
        self.cls = cls
        self.description = description
        self.example = example

    def __repr__ (self):
        if self.cls:
            return "<DocMethod '%s.%s'>" % (self.cls, self.name)
        return "<DocMethod '%s'>" % (self.name)

class Doc(object):
    def __init__ (self, filename):
        self.filename = filename
        self.showhead = True
        self.modulename = None
        self.modulealias = None
        self.shortdesc = "TODO"
        self.description = "TODO"
        self.example = ""
        self.classes = []
        self.data = {}
        self.functions = []
        self.includes = []
    
    def parse_content (self):
        dom = parse (self.filename)
        module = self.get_module_docs (dom)
        self.get_module_funcs (module)
        self.get_class_refs (module)
        self.get_data (module)
        self.get_includes (module)

    def get_data (self, module):
        data = module.getElementsByTagName ("data")
        for entry in data:
            name = entry.getAttribute ("name")
            val = entry.firstChild.nodeValue.strip ()
            self.data[name] = val

    def get_includes (self, module):
        incs = module.getElementsByTagName ("include")
        for node in incs:
            self.includes.append (node.firstChild.nodeValue)
    
    def get_module_docs (self, dom):
        module = dom.getElementsByTagName ("module")[0]
        self.modulename = module.getAttribute ("name")
        node = module.getElementsByTagName ("show")
        if node and node[0].firstChild.nodeValue:
            self.showhead = node[0].firstChild.nodeValue == "1"
        node = module.getElementsByTagName ("alias")
        if node and node[0].firstChild:
            self.modulealias = node[0].firstChild.nodeValue
            self.modulealias = self.modulealias.strip ()

        node = module.getElementsByTagName ("short")[0]
        if node.firstChild:
            self.shortdesc = node.firstChild.nodeValue
            self.shortdesc = self.shortdesc.strip ()
        node = module.getElementsByTagName ("desc")[0]
        if node.firstChild:
            self.description = node.firstChild.nodeValue
        node = module.getElementsByTagName ("example")
        if node and node[0].parentNode == module and node[0].firstChild:
            self.example = node[0].firstChild.nodeValue
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
            node = func.getElementsByTagName ("example")
            if node and node[0].firstChild:
                example = node[0].firstChild.nodeValue
            else:
                example = ""
            self.functions.append (DocMethod (None, name, call, desc, example))

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

            node = cls.getElementsByTagName ("example")
            if node and node[0].firstChild:
                example = node[0].firstChild.nodeValue
            else:
                example = ""
            clsdoc = DocClass (name, const, desc, example)

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
        node = attr.getElementsByTagName ("desc")[0]
        if node.firstChild:
            desc = node.firstChild.nodeValue
        else:
            desc = "TODO"
        node = attr.getElementsByTagName ("example")
        if node and node[0].firstChild:
            example = node[0].firstChild.nodeValue
        else:
            example = ""
        doccls.attributes.append (DocAttribute (name, desc, example))

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
        node = method.getElementsByTagName ("example")
        if node and node[0].firstChild:
            example = node[0].firstChild.nodeValue
        else:
            example = ""
        doccls.methods.append (DocMethod (doccls, name, call, desc, example))

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

    def create_example_rst (self, example, showex=False):
        data = ""
        if showex:
            data = "  **Example:** ::\n"
        else:
            data = "  ::\n"

        cindent = -1
        for line in example.split ("\n"):
            line = line.rstrip ()

            if cindent < 0:
                if len (line) != 0:
                    # Prepare indentation stuff
                    cindent = len (line) - len (line.lstrip ())

            if len (line) != 0:
                # Prepare indentation stuff
                indent = len (line) - len (line.lstrip ())
                line = line.lstrip ()
                if indent > cindent:
                    line = " " * (indent - cindent) + line

            data += "    " + line + "\n"
        return data + "\n"

    def create_rst (self):
        fname = os.path.join ("ref", "%s.rst")
        fp = open (fname % self.modulename.replace (".", "_"), "w")
        fp.write (RST_HEADER)

        name = self.modulename
        if self.modulealias:
            name = self.modulealias

        if self.showhead:
            fp.write (":mod:`%s` -- %s\n" % (name, self.shortdesc))
            fp.write ("%s\n" % ("=" * (11 + len (name) + len (self.shortdesc))))
            fp.write ("%s" % self.create_desc_rst (self.description))
            fp.write (".. module:: %s\n" % (name))
            fp.write ("   :synopsis: %s\n\n" % (self.shortdesc))
        else:
            fp.write (".. currentmodule:: %s\n\n" % (name))

        if len (self.example) > 0:
            fp.write (self.create_example_rst (self.example, True))

        if len (self.data) > 0:
            fp.write ("Data Fields\n")
            fp.write ("-----------\n")
            for key in self.data.keys ():
                fp.write (".. data:: %s\n" % key)
                fp.write (self.create_desc_rst (self.data[key], 2))

        if len (self.functions) > 0:
            fp.write ("Module Functions\n")
            fp.write ("----------------\n")
            for func in self.functions:
                fp.write (self.create_func_rst (func))
                if len (func.example) > 0:
                    fp.write (self.create_example_rst (func.example))

        if len (self.classes) > 0:
            for cls in self.classes:
                fp.write (cls.name + "\n")
                fp.write ("%s\n" % ("-" * len (cls.name)))
                fp.write (".. class:: %s\n" % cls.constructor)
                fp.write (self.create_desc_rst (cls.description, 2))
                if len (cls.example) > 0:
                    fp.write (self.create_example_rst (cls.example, True))
                if len (cls.attributes) > 0:
                        fp.write ("Attributes\n")
                        fp.write ("^^^^^^^^^^\n")
                        for attr in cls.attributes:
                            fp.write (".. attribute:: %s.%s\n"
                                      % (cls.name, attr.name))
                            fp.write (self.create_desc_rst \
                                      (attr.description, 2))
                            if len (attr.example) > 0:
                                fp.write ("Example:\n")
                                fp.write (self.create_example_rst \
                                          (attr.example))
                if len (cls.methods) > 0:
                    fp.write ("Methods\n")
                    fp.write ("^^^^^^^\n")
                    for method in cls.methods:
                        fp.write (".. method:: %s.%s\n" %
                                  (cls.name, method.calls[0]))
                        for call in method.calls[1:]:
                            fp.write ("            %s.%s\n" % (cls.name, call))
                        fp.write (self.create_desc_rst (method.description, 2))
                        if len (method.example) > 0:
                            fp.write ("Example:\n")
                            fp.write (self.create_example_rst (method.example))

        fp.write ("\n")
        for include in self.includes:
            fp.write (".. include:: %s\n" % include)
        fp.write (RST_FOOTER)
        fp.close ()
        
def create_rst (docs):
    if not os.path.exists ("ref"):
        os.mkdir ("ref")
    for doc in docs:
        print ("Now writing RST for %s...." % doc.modulename)
        doc.create_rst ()

def get_xml_files ():
    docs = []
    files = glob.glob (os.path.join ("src", "*.xml"))
    for fname in files:
        docs.append (Doc (fname))
    return docs

def get_rst_files ():
    return glob.glob (os.path.join ("src", "*.rst"))

if __name__ == "__main__":
    docs = get_xml_files ()
    for doc in docs:
        print ("Parsing file %s..." % doc.filename)
        doc.parse_content ()
    create_rst (docs)

    # Simply copy all other rst files
    docs = get_rst_files ()
    if not os.path.exists ("ref"):
        os.mkdir ("ref")
    for doc in docs:
        fname = os.path.basename (doc)
        shutil.copyfile (doc, os.path.join ("ref", "pygame2_%s" % fname))
