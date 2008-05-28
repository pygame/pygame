############################ PYGAME UNITTEST STUBBER ###########################
"""

README
======

Tests at gen_stubs_test.py

TODO
====

* more tests
* optparse command line settings

* only create needed stubs; untested units
    * naming convention for tests
        * name parser -- breakdown $Class, $unit, $description

* create class XXX(unittest.TestCase) stubs
    * relative indentation templates
        * find existing or implement

* output stubs as file per module

"""

from __future__ import with_statement
from collections import defaultdict

def get_methods_parent_class(f):
    #TODO: test this
    
    if hasattr(f, '__objclass__'):
        return f.__objclass__.__name__

def is_module(m):
    return m.__class__.__name__ == 'module'

def is_public(obj_name):
    #TODO: __init__ etc, will want to test constructors of ClassX
    
    return not obj_name.startswith(("__",))

def module_in_package(module, pkg):
    return ("%s." % pkg.__name__) in module.__name__

def get_package_modules(pkg):
    modules = (getattr(pkg, x) for x in dir(pkg))
    return [m for m in modules if is_module(m) and module_in_package(m, pkg)]
                                                 # Don't want to pick up 
                                                 # string module for example
                              
def get_doc_str(f, t=lambda x: x, default = ''):
    doc = f.__doc__
    if doc:
        if callable(t):
            return t(doc)
        else:
            return t % doc
    else:
        return default

def test_stub(f):
    parent_class = get_methods_parent_class(f)

    if parent_class:
        name = "%s__%s" % (parent_class, f.__name__)
    else:
        name = f.__name__

    doc_tmpl8 = '# Docstring:\n\n    # %s\n\n'
    doc = get_doc_str(f, lambda doc: doc_tmpl8 % doc.replace('\n', '\n    # '))

    test_doc = 'TODO: Test for unit, %s' % f.__name__

    stub =     ("def test_%s(self):\n"
                '    """\n    %s\n\n    """\n\n'
                '    %s\n\n'
                '    self.assert_(not_completed())'  
                '\n'% (name, test_doc, doc) )

    return name, stub

def get_callables(obj):
    publics = (getattr(obj, x) for x in dir(obj) if is_public(x))
    callables = [x for x in publics if callable(x)]
    return callables

def module_test_stubs(module):
    # get module.callable
    for one_deep in get_callables(module):
        yield test_stub(one_deep)
        
        # get module.Class.callable
        for two_deep in get_callables(one_deep):
            yield test_stub(two_deep)

def categorized_stubs(package):
    stubs_categorized = defaultdict(dict)

    for module in get_package_modules(package):
        for callable_testname_suffix, stub in module_test_stubs(module):
            stubs_categorized[module.__name__][callable_testname_suffix] = stub

    return stubs_categorized

def list_stubs(categorized, list_of = dict.keys):
    divider_line = '\n# ' + (78 * '*')
    module_heading_line = '# %s\n#\n'

    output = []
    for module, mapping in categorized.iteritems():
        output += [divider_line]
        output += [module_heading_line % module]

        # list_of:
        #     Use dict.keys for a listing of testname_suffixes one per line
        #         dict.values for a listing of stubs

        output += sorted(list_of(mapping))

    return output

if __name__ == "__main__":
    import pygame, sys
    pygame.init()

    stubs = categorized_stubs(pygame)

    # TODO: optparse: stubs per module, run tests etc
    if 'list' in sys.argv: render_dict_part = dict.keys
    else: render_dict_part = dict.values

    with open('test_stubs.py', 'w') as fh:
        fh.write( "\n".join (
            list_stubs(stubs, list_of = render_dict_part)
        ))

    total_stubs, total_modules = sum(len(m) for m in stubs), len(stubs)

    print 'Done: %s stubs created from %s modules containing callables' %\
            (total_stubs,          total_modules)

################################################################################