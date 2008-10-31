from xml.dom.minidom import parse
import os, glob, sys

HTML_HEAD = """
<html>
<title>%s - Pygame Documentation</title>
<body bgcolor=#aaeebb text=#000000 link=#331111 vlink=#331111>

<table cellpadding=0 cellspacing=0 border=0 style='border: 3px solid black;' width='100%%'>
<tr>
<td bgcolor='#c2fc20' style='padding: 6px;' align=center valign=center><a href='http://www.pygame.org/'><img src='../pygame_tiny.gif' border=0 width=200 height=60></a><br><b>pygame documentation</b></td>
<td bgcolor='#6aee28' style='border-left: 3px solid black; padding: 6px;' align=center valign=center>
	||&nbsp;
	<a href=http://www.pygame.org>Pygame Home</a> &nbsp;||&nbsp;
	<a href=../index.html>Help Contents</a> &nbsp;||
	<a href=index.html>Reference Index</a> &nbsp;||
	<br>&nbsp;<br>
	
"""

class DocClass (object):
    def __init__ (self, name, description):
        self.name = name
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
    def __init__ (self, cls, name, description):
        self.name = name
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
        self.description = None
        self.classes = []
        self.functions = []
    
    def parse_content (self):
        dom = parse (self.filename)
        module = self.get_module_docs (dom)
        self.get_module_funcs (module)
        self.get_class_refs (module)
    
    def get_module_docs (self, dom):
        module = dom.getElementsByTagName ("module")[0]
        self.module = module.getAttribute ("name")
        self.description = module.getElementsByTagName \
            ("desc")[0].firstChild.nodeValue
        return module
    
    def get_module_funcs (self, module):
        functions = module.getElementsByTagName ("func")
        for func in functions:
            name = func.getAttribute ("name")
            desc = func.getElementsByTagName ("desc")[0].firstChild.nodeValue
            self.functions.append (DocMethod (None, name, desc))

    def get_class_refs (self, module):
        classes = module.getElementsByTagName ("class")
        for cls in classes:
            name = cls.getAttribute ("name")
            desc = cls.getElementsByTagName ("desc")[0].firstChild.nodeValue
            clsdoc = DocClass (name, desc)

            attrs = cls.getElementsByTagName ("attr")
            for attr in attrs:
                self.create_attr_ref (clsdoc, attr)

            methods = cls.getElementsByTagName ("method")
            for method in methods:
                self.create_method_ref (clsdoc, method)
            
            self.classes.append (clsdoc)

    def create_attr_ref (self, doccls, attr):
        name = attr.getAttribute ("name")
        desc = attr.firstChild.nodeValue
        doccls.attributes.append (DocAttribute (name, desc))

    def create_method_ref (self, doccls, method):
        name = method.getAttribute ("name")
        desc = method.getElementsByTagName ("desc")[0].firstChild.nodeValue
        doccls.methods.append (DocMethod (doccls, name, desc))
        
    def create_html (self, doclist):
        # TODO
        pass

def get_doc_files ():
    docs = []
    files = glob.glob (os.path.join ("src", "*.xml"))
    for fname in files:
        docs.append (Doc (fname))
    return docs

if __name__ == "__main__":
    docs = get_doc_files ()
    for doc in docs:
        doc.parse_content ()
        doc.create_html (docs)
