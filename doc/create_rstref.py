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
        self.call = call
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
        self.description = ""
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
        node = module.getElementsByTagName ("desc")[0]
        if node.firstChild:
            self.description = node.firstChild.nodeValue
        return module
    
    def get_module_funcs (self, module):
        functions = module.getElementsByTagName ("func")
        for func in functions:
            name = func.getAttribute ("name")
            node = func.getElementsByTagName ("call")[0]
            if node.firstChild:
                call = node.firstChild.nodeValue
            else:
                call = ""
            node = func.getElementsByTagName ("desc")[0]
            if node.firstChild:
                desc = node.firstChild.nodeValue
            else:
                desc = ""
            self.functions.append (DocMethod (None, name, call, desc))

    def get_class_refs (self, module):
        classes = module.getElementsByTagName ("class")
        for cls in classes:
            name = cls.getAttribute ("name")
            node = cls.getElementsByTagName ("constructor")[0]
            if node.firstChild:
                const = node.firstChild.nodeValue
            else:
                const = ""
            node = cls.getElementsByTagName ("desc")[0]
            if node.firstChild:
                desc = node.firstChild.nodeValue
            else:
                desc = ""
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
        if attr.firstChild:
            desc = attr.firstChild.nodeValue
        else:
            desc = ""
        doccls.attributes.append (DocAttribute (name, desc))

    def create_method_ref (self, doccls, method):
        name = method.getAttribute ("name")
        node = method.getElementsByTagName ("call")[0]
        if node.firstChild:
            call = node.firstChild.nodeValue
        else:
            call = ""
        node = method.getElementsByTagName ("desc")[0]
        if node.firstChild:
            desc = node.firstChild.nodeValue
        else:
            desc = ""
        doccls.methods.append (DocMethod (doccls, name, call, desc))

    def create_desc_rst (self, desc, refcache):
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
                data += "\n" + line
                written += 1
        if written == 0:
            return ""
        return data + "\n\n"
    
    def create_func_rst (self, func, refcache):
        data = "**%s**\n" % (func.name)
        data += "*%s*\n" % (func.call)
        data += "%s\n" % self.create_desc_rst (func.description, refcache)
        return data
    
    def create_rst (self, refcache):
        fname = os.path.join ("ref", "%s.rst")
        fp = open (fname % self.modulename.replace (".", "_"), "w")
        fp.write (RST_HEADER)
        
        fp.write ("%s\n" % self.modulename)
        fp.write ("=" * len (self.modulename) + "\n")
        fp.write ("%s\n\n" % self.create_desc_rst (self.description, refcache))
        
        if len (self.functions) > 0:
            fp.write ("\n")
            fp.write ("Module functions\n")
            fp.write ("----------------\n")
            for func in self.functions:
                fp.write (self.create_func_rst (func, refcache))

        if len (self.classes) > 0:
            for cls in self.classes:
                fp.write (cls.name + "\n")
                fp.write ("-" * len (cls.name) + "\n")
                fp.write ("*" + cls.constructor + "*\n")
                fp.write (self.create_desc_rst (cls.description, refcache))
                if len (cls.attributes) > 0:
                        fp.write ("Attributes\n")
                        fp.write ("^^^^^^^^^^\n")
                        for attr in cls.attributes:
                            fp.write ("**" + attr.name + "**")
                            fp.write (self.create_desc_rst (attr.description,
                                                            refcache))
                if len (cls.methods) > 0:
                    fp.write ("Methods\n")
                    fp.write ("^^^^^^^\n")
                    for method in cls.methods:
                        fp.write ("**" + method.name + "**\n")
                        fp.write ("*" + method.call + "*\n")
                        fp.write (self.create_desc_rst (method.description,
                                                        refcache))
        fp.write ("\n")
        fp.write (RST_FOOTER)
        fp.close ()
        
def create_rst (docs):
    refcache = {}
    if not os.path.exists ("ref"):
        os.mkdir ("ref")

    for doc in docs:
        for cls in doc.classes:
            refcache[cls.name] = doc.filename
        for func in doc.functions:
            refcache[func.name] = doc.filename
    for doc in docs:
        print ("Now writing RST for %s...." % doc.modulename)
        doc.create_rst (refcache)

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
