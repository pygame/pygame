import docutils.nodes
import sphinx.addnodes


class GetError(LookupError):
    pass

def get_fullname(node):
    if isinstance(node, docutils.nodes.section):
        return get_sectionname(node)
    if isinstance(node, sphinx.addnodes.desc):
        return get_descname(node)
    raise TypeError("Unrecognized node type '%s'" % (node.__class__,))

def get_descname(desc):
    try:
        sig = desc[0]
    except IndexError:
        raise GetError("No fullname: missing children in desc")
    try:
        names = sig['names']
    except KeyError:
        raise GetError(
            "No fullname: missing names attribute in desc's child")
    try:
        return names[0]
    except IndexError:
        raise GetError("No fullname: desc's child has empty names list")

def get_sectionname(section):
    try:
        names = section['names']
    except KeyError:
        raise GetError("No fullname: missing names attribute in section")
    try:
        return names[0]
    except IndexError:
        raise GetError("No fullname: section has empty names list")

def get_refuri(node):
    return as_refuri(get_refid(node))

def get_refid(node):
    try:
        return get_ids(node)[0]
    except IndexError:
        raise GetError("Node has emtpy ids list")

def as_refid(refuri):
    return refuri[1:]

def as_refuri(refid):
    return NUMBERSIGN + refid

def get_ids(node):
    if isinstance(node, docutils.nodes.section):
        try:
            return node['ids']
        except KeyError:
            raise GetError("No ids: section missing ids attribute")
    if isinstance(node, sphinx.addnodes.desc):
        try:
            sig = node[0]
        except IndexError:
            raise GetError("No ids: missing desc children")
        try:
            return sig['ids']
        except KeyError:
            raise GetError("No ids: desc's child missing ids attribute")
    raise TypeError("Unrecognized node type '%s'" % (node.__class__,))

def isections(doctree):
    for node in doctree:
        if isinstance(node, docutils.nodes.section):
            yield node

def get_name(fullname):
    return fullname.split('.')[-1]

def geterror ():
    return sys.exc_info()[1]

try:
    _unicode = unicode
except NameError:
    _unicode = str

# Represent escaped bytes and strings in a portable way.
#
# as_bytes: Allow a Python 3.x string to represent a bytes object.
#   e.g.: as_bytes("a\x01\b") == b"a\x01b" # Python 3.x
#         as_bytes("a\x01\b") == "a\x01b"  # Python 2.x
# as_unicode: Allow a Python "r" string to represent a unicode string.
#   e.g.: as_unicode(r"Bo\u00F6tes") == u"Bo\u00F6tes" # Python 2.x
#         as_unicode(r"Bo\u00F6tes") == "Bo\u00F6tes"  # Python 3.x
try:
    eval("u'a'")
    def as_bytes(string):
        """ '<binary literal>' => '<binary literal>' """
        return string
        
    def as_unicode(rstring):
        """ r'<Unicode literal>' => u'<Unicode literal>' """
        return rstring.decode('unicode_escape', 'strict')
        
except SyntaxError:
    def as_bytes(string):
        """ '<binary literal>' => b'<binary literal>' """
        return string.encode('latin-1', 'strict')
        
    def as_unicode(rstring):
        """ r'<Unicode literal>' => '<Unicode literal>' """
        return rstring.encode('ascii', 'strict').decode('unicode_escape',
                                                        'stict')

# Ensure Visitor is a new-style class
_SparseNodeVisitor = docutils.nodes.SparseNodeVisitor
if not hasattr(_SparseNodeVisitor, '__class__'):
    class _SparseNodeVisitor(object, docutils.nodes.SparseNodeVisitor):
        pass

class Visitor(_SparseNodeVisitor):

    skip_node = docutils.nodes.SkipNode()
    skip_departure = docutils.nodes.SkipDeparture()

    def __init__(self, app, document_node):
        docutils.nodes.SparseNodeVisitor.__init__(self, document_node)
        self.app = app
        self.env = app.builder.env

    def unknown_visit(self, node):
        return

    def unknown_departure(self, node):
        return

EMPTYSTR = as_unicode('')
NUMBERSIGN = as_unicode('#')
