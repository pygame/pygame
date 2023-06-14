"""Collect information on document sections and Pygame API objects

The following persistent Pygame specific environment structures are built:

pyg_sections: [{'docname': <docname>,
                'fullname': <fullname>,
                'refid': <ref>},
               ...]
    all Pygame api sections in the documents in order processed.
pyg_descinfo_tbl: {<id>: {'fullname': <fullname>,
                          'desctype': <type>,
                          'summary': <summary>,
                          'signatures': <sigs>,
                          'children': <toc>,
                          'refid': <ref>,
                          'docname': <docname>},
                   ...}
    object specific information, including a list of direct children, if any.

<docname>: (str) the simple document name without path or extension.
<fullname>: (str) a fully qualified object name. It is a unique identifier.
<ref>: (str) an id usable as local uri reference.
<id>: (str) unique desc id, the first entry in the ids attribute list.
<type>: (str) an object's type: the desctype attribute.
<summary>: (str) a summary line as identified by a :summaryline: role.
           This corresponds to the first line of a docstring.
<sigs>: (list of str) an object's signatures, in document order.
<toc>: (list of str) refids of an object's children, in document order.

"""

from ext.utils import Visitor, get_fullname, get_refid, as_refid, GetError

from collections import deque

import os.path

MODULE_ID_PREFIX = "module-"


def setup(app):
    app.connect("env-purge-doc", prep_document_info)
    app.connect("doctree-read", collect_document_info)


def prep_document_info(app, env, docname):
    try:
        env.pyg_sections = [e for e in env.pyg_sections if e["docname"] != docname]
    except AttributeError:
        pass
    except KeyError:
        pass
    try:
        descinfo_tbl = env.pyg_descinfo_tbl
    except AttributeError:
        pass
    else:
        to_remove = [k for k, v in descinfo_tbl.items() if v["docname"] == docname]
        for k in to_remove:
            del descinfo_tbl[k]


def collect_document_info(app, doctree):
    doctree.walkabout(CollectInfo(app, doctree))


class CollectInfo(Visitor):

    """Records the information for a document"""

    desctypes = {
        "data",
        "function",
        "exception",
        "class",
        "attribute",
        "method",
        "staticmethod",
        "classmethod",
    }

    def __init__(self, app, document_node):
        super().__init__(app, document_node)
        self.docname = self.env.docname
        self.summary_stack = deque()
        self.sig_stack = deque()
        self.desc_stack = deque()
        try:
            self.env.pyg_sections
        except AttributeError:
            self.env.pyg_sections = []
        try:
            self.env.pyg_descinfo_tbl
        except AttributeError:
            self.env.pyg_descinfo_tbl = {}

    def visit_document(self, node):
        # Only index pygame Python API documents, found in the docs/reST/ref
        # subdirectory. Thus the tutorials and the C API documents are skipped.
        source = node["source"]
        head, file_name = os.path.split(source)
        if not file_name:
            raise self.skip_node
        head, dir_name = os.path.split(head)
        if not (dir_name == "ref" or dir_name == "referencias"):
            raise self.skip_node
        head, dir_name = os.path.split(head)
        if not (dir_name == "reST" or dir_name == "es"):
            raise self.skip_node
        head, dir_name = os.path.split(head)
        if dir_name != "docs":
            raise self.skip_node

    def visit_section(self, node):
        if not node["names"]:
            raise self.skip_node
        self._push()

    def depart_section(self, node):
        """Record section info"""

        summary, sigs, child_descs = self._pop()
        if not node.children:
            return
        if node["ids"][0].startswith(MODULE_ID_PREFIX):
            self._add_section(node)
            self._add_descinfo(node, summary, sigs, child_descs)
        elif child_descs:
            # No section level introduction: use the first toplevel directive
            # instead.
            desc_node = child_descs[0]
            summary = get_descinfo(desc_node, self.env).get("summary", "")
            self._add_section(desc_node)
            self._add_descinfo_entry(node, get_descinfo(desc_node, self.env))

    def visit_desc(self, node):
        """Prepare to collect a summary and toc for this description"""

        if node.get("desctype", "") not in self.desctypes:
            raise self.skip_node
        self._push()

    def depart_desc(self, node):
        """Record descinfo information and add descinfo to parent's toc"""

        self._add_descinfo(node, *self._pop())
        self._add_desc(node)

    def visit_inline(self, node):
        """Collect a summary or signature"""

        if "summaryline" in node["classes"]:
            self._add_summary(node)
        elif "signature" in node["classes"]:
            self._add_sig(node)
        raise self.skip_departure

    def _add_section(self, node):
        entry = {
            "docname": self.docname,
            "fullname": get_fullname(node),
            "refid": get_refid(node),
        }
        self.env.pyg_sections.append(entry)

    def _add_descinfo(self, node, summary, sigs, child_descs):
        entry = {
            "fullname": get_fullname(node),
            "desctype": node.get("desctype", "module"),
            "summary": summary,
            "signatures": sigs,
            "children": [get_refid(n) for n in child_descs],
            "refid": get_refid(node),
            "docname": self.docname,
        }
        self._add_descinfo_entry(node, entry)

    def _add_descinfo_entry(self, node, entry):
        key = get_refid(node)
        if key.startswith(MODULE_ID_PREFIX):
            key = key[len(MODULE_ID_PREFIX) :]
        self.env.pyg_descinfo_tbl[key] = entry

    def _push(self):
        self.summary_stack.append("")
        self.sig_stack.append([])
        self.desc_stack.append([])

    def _pop(self):
        return (self.summary_stack.pop(), self.sig_stack.pop(), self.desc_stack.pop())

    def _add_desc(self, desc_node):
        self.desc_stack[-1].append(desc_node)

    def _add_summary(self, text_element_node):
        self.summary_stack[-1] = text_element_node[0].astext()

    def _add_sig(self, text_element_node):
        self.sig_stack[-1].append(text_element_node[0].astext())


def tour_descinfo(fn, node, env):
    try:
        descinfo = get_descinfo(node, env)
    except GetError:
        return
    fn(descinfo)
    for refid in descinfo["children"]:
        tour_descinfo_refid(fn, refid, env)


def tour_descinfo_refid(fn, refid, env):
    descinfo = env.pyg_descinfo_tbl[refid]  # A KeyError would mean a bug.
    fn(descinfo)
    for refid in descinfo["children"]:
        tour_descinfo_refid(fn, refid, env)


def get_descinfo(node, env):
    return get_descinfo_refid(get_refid(node), env)


def get_descinfo_refid(refid, env):
    if refid.startswith(MODULE_ID_PREFIX):
        refid = refid[len(MODULE_ID_PREFIX) :]
    try:
        return env.pyg_descinfo_tbl[refid]
    except KeyError:
        raise GetError("Not found")
