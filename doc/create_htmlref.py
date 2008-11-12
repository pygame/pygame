from xml.dom.minidom import parse
import os, glob, sys

HTML_HEADER = """
<html>
<head>
<title>%s</title>
</head>
<body>
"""

HTML_FOOTER = """
</body>
</html>
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
            node = func.getElementsByTagName ("desc")[0]
            if node.firstChild:
                desc = node.firstChild.nodeValue
            else:
                desc = ""
            self.functions.append (DocMethod (None, name, desc))

    def get_class_refs (self, module):
        classes = module.getElementsByTagName ("class")
        for cls in classes:
            name = cls.getAttribute ("name")
            node = cls.getElementsByTagName ("desc")[0]
            if node.firstChild:
                desc = node.firstChild.nodeValue
            else:
                desc = ""
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
        if attr.firstChild:
            desc = attr.firstChild.nodeValue
        else:
            desc = ""
        doccls.attributes.append (DocAttribute (name, desc))

    def create_method_ref (self, doccls, method):
        name = method.getAttribute ("name")
        node = method.getElementsByTagName ("desc")[0]
        if node.firstChild:
            desc = node.firstChild.nodeValue
        else:
            desc = ""
        doccls.methods.append (DocMethod (doccls, name, desc))

    def create_desc_html (self, desc, refcache):
        written = 0
        blocks = 0
        data = '<p>'
        for line in desc.split ('\n'):
            line = line.strip ()
            if written > 0 and line == '':
                data += '</p><p>'
                blocks += 1
            elif line != '':
                data += line
                written += 1
        if blocks > 0:
            data += '</p>'
        if written == 0:
            return ''
        return data
    
    def create_func_html (self, func, refcache):
        data = '    <dt class="functions"><a name="%s">%s</a></dt>\n' % \
               (func.name, func.name)
        data += '    <dd class="functions">%s</dd>\n' % \
                self.create_desc_html (func.description, refcache)
        return data
    
    def create_html (self, refcache):
        fname = os.path.join ("ref", "%s.html")
        fp = open (fname % self.modulename.replace (".", "_"), "w")
        fp.write (HTML_HEADER % self.modulename)
        
        fp.write ('<h1 class="module">%s</h1>\n' % self.modulename)
        fp.write ('<div class="module">\n%s</div>\n' % self.description)
        fp.write ('<div class="moddefs">\n')
        
        if len (self.functions) > 0:
            fp.write ('  <dl class="functions">\n')
            for func in self.functions:
                fp.write (self.create_func_html (func, refcache))
            fp.write ("  </dl>\n")

        if len (self.classes) > 0:
            for cls in self.classes:
                fp.write (cls.name)
                fp.write (cls.description)
                for attr in cls.attributes:
                    fp.write (attr.name)
                    fp.write (attr.description)
                for method in cls.methods:
                    fp.write (method.name)
                    fp.write (method.description)
        fp.write ('</div>\n')
        fp.write (HTML_FOOTER)
        fp.close ()
        
def create_html (docs):
    refcache = {}
    if not os.path.exists ("ref"):
        os.mkdir ("ref")

    for doc in docs:
        for cls in doc.classes:
            refcache[cls.name] = doc.filename
        for func in doc.functions:
            refcache[func.name] = doc.filename
    for doc in docs:
        print ("Now writing HTML for %s...." % doc.modulename)
        doc.create_html (refcache)

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
    create_html (docs)
