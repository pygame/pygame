"""Add the generic fixed and derived content to a Classic Pygame document"""

from ext.utils import (Visitor, _unicode, as_unicode, get_name, GetError,
                       get_refid, as_refid, as_refuri)
from ext.indexer import get_descinfo, get_descinfo_refid

from sphinx.addnodes import (desc, desc_signature, desc_content, module)

from docutils.nodes import (section, literal, reference, paragraph, title,
                            Text, TextElement,
                            table, tgroup, colspec, tbody, row, entry,
                            whitespace_normalize_name, SkipNode)
import os
import re
from collections import deque

DOT = as_unicode('.')
SPACE = as_unicode(' ')
EMDASH = as_unicode(r'\u2014')

def setup(app): 
    # This extension uses indexer collected tables.
    app.setup_extension('ext.indexer')

    app.connect('doctree-resolved', transform_document)
    app.connect('html-page-context', inject_template_globals)
    app.add_node(TocRef,
                 html=(visit_toc_ref_html, depart_toc_ref_html),
                 latex=(visit_toc_ref, depart_toc_ref),
                 text=(visit_toc_ref, depart_toc_ref))
    app.add_node(TocTable,
                 html=(visit_toc_table_html, depart_toc_table_html),
                 latex=(visit_skip, None), text=(visit_skip, None))
    app.add_node(DocTitle,
                 html=(visit_doc_title_html, depart_doc_title_html),
                 latex=(visit_doc_title, depart_doc_title))


class TocRef(reference):
    pass

def visit_toc_ref(self, node):
    self.visit_reference(node)

def depart_toc_ref(self, node):    
    self.depart_reference(node)

def visit_toc_ref_html(self, node):
    refuri = node['refuri']
    refid = as_refid(refuri)
    docname = get_descinfo_refid(refid, self.settings.env)['docname']
    link_suffix = self.builder.link_suffix
    node['refuri'] = '%s%s%s' % (docname, link_suffix, refuri)
    visit_toc_ref(self, node)

class TocTable(table):
    pass

def visit_toc_table_html(self, node):
    self.visit_table(node)

def depart_toc_table_html(self, node):
    self.depart_table(node)

def visit_skip(self, node):
    raise SkipNode()

depart_toc_ref_html = depart_toc_ref

class DocTitle(title):
    pass

visit_doc_title_html = visit_skip
depart_doc_title_html = None

def visit_doc_title(self, node):
    self.visit_title(node)

def depart_doc_title(self, node):
    self.depart_title(node)

def transform_document(app, doctree, docname):
    doctree.walkabout(DocumentTransformer(app, doctree))

class DocumentTransformer(Visitor):
    _key_re = r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
    key_pat = re.compile(_key_re)
    
    def __init__(self, app, document_node):
        super(DocumentTransformer, self).__init__(app, document_node)
        self.module_stack = deque()
        self.title_stack = deque()

    def visit_section(self, node):
        self.module_stack.append(None)
        self.title_stack.append(None)

    def depart_section(self, node):
        module_node = self.module_stack.pop()
        title_node = self.title_stack.pop()
        if module_node is not None:
            transform_module_section(node, title_node, module_node, self.env)

    def visit_module(self, node):
        self.module_stack.append(node)

    def visit_desc(self, node):
        pass

    def depart_desc(self, node):
        node['classes'].append('definition')
        node[0]['classes'].append('title')
        add_toc(node, self.env)

    def visit_title(self, node):
        if isinstance(node.parent, section):
            # Make title node a DocTitle instance. This works because DocTitle
            # simply subclasses title.
            node.__class__ = DocTitle

    def visit_reference(self, node):
        if 'toc' in node['classes']:
            return
        try:
            child = node[0]
        except IndexError:
            return
        if not isinstance(child, TextElement):
            return
        name = child.astext()
        m = self.key_pat.match(name)
        if m is None:
            return
        key = m.group('key')
        try:
            summary = get_descinfo_refid(key, self.env)['summary']
        except GetError:
            return
        if summary:
            node['reftitle'] = summary

def transform_module_section(section_node, title_node, module_node, env):
    fullmodname = module_node['modname']
    where = section_node.index(module_node)
    content_children = list(ipop_child(section_node, where + 1))
    if title_node is None:
        signature_children = [literal('', fullmodname)]
    else:
        signature_children = list(ipop_child(title_node))
    signature_node = desc_signature('', '', *signature_children,
                                    classes=['title', 'module'],
                                    names=[fullmodname])
    content_node = desc_content('', *content_children)
    desc_node = desc('', signature_node, content_node,
                     desctype='module', classes=['definition'])
    section_node.append(desc_node)
    add_toc(desc_node, env, section_node)

def ipop_child(node, start=0):
    while len(node) > start:
        yield node.pop(start)

def get_target_summary(reference_node, env):
    try:
        return get_descinfo_refid(reference_node['refid'], env)['summary']
    except KeyError:
        raise GetError("reference has no refid")
    
def add_toc(desc_node, env, section_node=None):
    """Add a table of contents to a desc node"""

    if (section_node is not None):
        refid = get_refid(section_node)
    else:
        refid = get_refid(desc_node)
    descinfo = get_descinfo_refid(refid, env)
    toc = build_toc(descinfo, env)
    if toc is None:
        return
    content_node = desc_node[-1]
    insert_at = 0
    if descinfo['summary']:  # if have a summary
        insert_at += 1
    content_node.insert(insert_at, toc)

def build_toc(descinfo, env):
    """Return a desc table of contents node tree"""
    
    child_ids = descinfo['children']
    if not child_ids:
        return None
    max_fullname_len = 0
    max_summary_len = 0
    rows = []
    for fullname, refid, summary in ichild_ids(child_ids, env):
        max_fullname_len = max(max_fullname_len, len(fullname))
        max_summary_len = max(max_summary_len, len(summary))
        reference_node = toc_ref(fullname, refid)
        ref_entry_node = entry('', paragraph('', '', reference_node))
        sum_entry_node = entry('', paragraph('', EMDASH + SPACE + summary))
        row_node = row('', ref_entry_node, sum_entry_node)
        rows.append(row_node)
    col0_len = max_fullname_len + 2   # add error margin
    col1_len = max_summary_len + 12   # add error margin and room for emdash
    tbody_node = tbody('', *rows)
    col0_colspec_node = colspec(colwidth=col0_len)
    col1_colspec_node = colspec(colwidth=col1_len)
    tgroup_node = tgroup('', col0_colspec_node, col1_colspec_node, tbody_node,
                         cols=2)
    return TocTable('', tgroup_node, classes=['toc'])

def ichild_ids(child_ids, env):
    for refid in child_ids:
        descinfo = env.pyg_descinfo_tbl[refid]  # A KeyError would mean a bug.
        yield descinfo['fullname'], descinfo['refid'], descinfo['summary']

def toc_ref(fullname, refid):
    name = whitespace_normalize_name(fullname),
    return TocRef('', fullname,
                  name=name, refuri=as_refuri(refid), classes=['toc'])


#>>===================================================
def decorate_signatures(desc, classname):
    prefix = classname + DOT
    for child in desc.children:
        if (isinstance(child, sphinx.addnodes.desc_signature) and
            isinstance(child[0], sphinx.addnodes.desc_name)       ):
            new_desc_classname = sphinx.addnodes.desc_classname('', prefix)
            child.insert(0, new_desc_classname)

#<<==================================================================

def inject_template_globals(app, pagename, templatename, context, doctree):
    def lowercase_name(d):
        return get_name(d['fullname']).lower()

    env = app.builder.env
    try:
        sections = env.pyg_sections
    except AttributeError:
        sections = []
    else:
        sections = sorted(sections, key=lowercase_name)
    context['pyg_sections'] = sections
