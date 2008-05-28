############################ PYGAME UNITTEST STUBBER ###########################

"""

==========
= README =
==========

Tests at gen_stubs_test.py

TODO
====

* more tests
* optparse command line settings

* only create needed stubs; untested units
    * naming convention for tests
        * name parser -- breakdown $Class, $unit, $description

* create class XXX(unittest.TestCase) stubs
    * relative indentation templates                                    = DONE =
        * find existing or implement                                    = DONE =

* output stubs as file per module


BUGS
====

* needs tests


"""

#################################### IMPORTS ###################################

from __future__ import with_statement
from collections import defaultdict

import relative_indentation

################################ STUB TEMPLATES ################################

STUB_TEMPLATE = relative_indentation.Template ( '''

    def ${test_name}(self):

        """
            TODO: Test unit ${docstring}
        """

        # Doc string for unit ${unitname}:

            ${comments}

        self.assert_(not_completed() ''', strip_common = 1, strip_excess = 1
)

# for list_all_stubs

DIVIDER_LINE = '\n# ' + (78 * '*')

MODULE_HEADING_LINE = '# %s\n#\n'

################################### FUNCTIONS ##################################

def get_methods_parent_class(f):
    #TODO: test this
    if hasattr(f, 'im_class'):
        return f.im_class.__name__
    
    elif hasattr(f, '__objclass__'):
        return f.__objclass__.__name__

def is_module(m):
    return m.__class__.__name__ == 'module'

def is_public(obj_name):
    #TODO: __init__ etc, will want to test constructors of Class X

    return not obj_name.startswith(('__','_'))

def module_in_package(module, pkg):
    return ("%s." % pkg.__name__) in module.__name__

def get_package_modules(pkg):
    modules = (getattr(pkg, x) for x in dir(pkg))
    return [m for m in modules if is_module(m) and module_in_package(m, pkg)]
                                                 # Don't want to pick up 
                                                 # string module for example

def get_doc_str(f, t = None, default = ''):
    doc = f.__doc__
    if doc:
        if not t:
            return doc
        elif callable(t):
            return t(doc)
        else:
            return t % doc
    else:
        return default

def py_comment(input_str):
    return '\n'.join([('# ' + l.strip(' ')) for l in input_str.split('\n')])

################################################################################

def test_stub(f, module):
    parent_class = get_methods_parent_class(f)

    test_name = 'test_'
    unit_name = '%s.' % module.__name__

    if parent_class:
        unit_name += '%s.' % parent_class
        test_name += "%s__" % parent_class

    test_name += f.__name__
    unit_name += f.__name__

    stub = STUB_TEMPLATE.render (

            test_name = test_name,
            comments = py_comment(get_doc_str(f)),
            docstring = unit_name,
            unitname = unit_name,
    )

    return test_name, stub

################################################################################

def get_callables(obj):
    publics = (getattr(obj, x) for x in dir(obj) if is_public(x))
    callables = [x for x in publics if callable(x)]
    return callables

def callables_from_module(m):
    #TODO: look into what functions don't have __module__ attribute
    return filter (
        lambda f: not hasattr(f, '__module__') or f.__module__ == m.__name__,
        get_callables(m)
    )

def module_test_stubs(m):
    # get module.callable
    for one_deep in callables_from_module(m):
        yield test_stub(one_deep, m)

        # get module.Class.callable
        for two_deep in get_callables(one_deep):
            yield test_stub(two_deep, m)

def categorized_stubs(package):
    stubs_categorized = defaultdict(dict)

    for module in [package] + get_package_modules(package):
        for callable_testname, stub in module_test_stubs(module):
            stubs_categorized[module.__name__][callable_testname] = stub

    return stubs_categorized

def list_all_stubs(categorized, list_of = dict.keys):
    output = []
    for module, mapping in categorized.iteritems():
        output += [DIVIDER_LINE]
        output += [MODULE_HEADING_LINE % module]

        # list_of:
        #     Use dict.keys for a listing of testname_suffixes one per line
        #         dict.values for a listing of stubs

        output += sorted(list_of(mapping))

    return '\n'.join(output)

if __name__ == "__main__":
    import pygame, sys
    # pygame.init()                                    # same numbers as without 

    stubs = categorized_stubs(pygame)

    # TODO: optparse: stubs per module, run tests etc
    if 'list' in sys.argv: render_dict_part = dict.keys
    else: render_dict_part = dict.values

    with open('test_stubs.py', 'w') as fh:
        fh.write( list_all_stubs(stubs, list_of = render_dict_part) )

    total_stubs  = sum(len(m) for _, m in stubs.iteritems())
    total_modules = len(stubs)

    print 'Done: %s public callable stubs created from %s modules containing' %\
            (total_stubs,                          total_modules)

################################################################################