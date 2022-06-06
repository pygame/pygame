import docutils.nodes
import sphinx.addnodes

import sys


class GetError(LookupError):
    pass


def get_fullname(node):
    if isinstance(node, docutils.nodes.section):
        return get_sectionname(node)
    if isinstance(node, sphinx.addnodes.desc):
        return get_descname(node)
    raise TypeError(f"Unrecognized node type '{node.__class__}'")


def get_descname(desc):
    try:
        sig = desc[0]
    except IndexError:
        raise GetError("No fullname: missing children in desc")
    try:
        names = sig["ids"]
    except KeyError:
        raise GetError("No fullname: missing ids attribute in desc's child")
    try:
        return names[0]
    except IndexError:
        raise GetError("No fullname: desc's child has empty names list")


def get_sectionname(section):
    try:
        names = section["names"]
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
        raise GetError("Node has empty ids list")


def as_refid(refuri):
    return refuri[1:]


def as_refuri(refid):
    return "#" + refid


def get_ids(node):
    if isinstance(node, docutils.nodes.section):
        try:
            return node["ids"]
        except KeyError:
            raise GetError("No ids: section missing ids attribute")
    if isinstance(node, sphinx.addnodes.desc):
        try:
            sig = node[0]
        except IndexError:
            raise GetError("No ids: missing desc children")
        try:
            return sig["ids"]
        except KeyError:
            raise GetError("No ids: desc's child missing ids attribute")
    raise TypeError(f"Unrecognized node type '{node.__class__}'")


def isections(doctree):
    for node in doctree:
        if isinstance(node, docutils.nodes.section):
            yield node


def get_name(fullname):
    return fullname.split(".")[-1]


class Visitor(docutils.nodes.SparseNodeVisitor):
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
