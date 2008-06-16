#################################### IMPORTS ###################################

from __future__ import with_statement

from optparse import OptionParser

from inspect import isclass, ismodule, getdoc

from unittest import TestCase

import pygame, sys, relative_indentation, re

################################ TESTS DIRECTORY ###############################

from os.path import normpath, join, dirname, abspath

sys.path.append(
    abspath(normpath(
        join(dirname(__file__), '../')
    ))
)

#################################### IGNORES ###################################

# pygame.sprite.Sprite.__module__ = 'pygame.sprite' 
# pygame.sprite.Rect.__module__   = 'pygame'

# Can not then filter out nonsensical classes automatically

IGNORE = (
    'pygame.sprite.Rect',
)

##################################### TODO #####################################

"""

1)
    IGNORE. Find a better solution.


2)
    Optimize:
        eg.
            $  gen_stubs.py sprite.Sprite 
              only stub out sprite module etc

3)

    Test:

"""

################################ STUB TEMPLATES ################################

STUB_TEMPLATE = relative_indentation.Template ( '''

    def ${test_name}(self):

        # Doc string for ${unitname}:

          ${comments}

        self.assert_(test_not_implemented()) ''', 

        strip_common = 1, strip_excess = 1
)


############################## REGULAR EXPRESSIONS #############################

module_re = re.compile(r"pygame\.([^.]*)\.?")

#################################### OPTIONS ###################################

opt_parser = OptionParser()

opt_parser.add_option(

     "-l",  "--list",
     dest   = "list",
     action = 'store_true',
     help   = "list only test names not stubs" )

opt_parser.set_usage(
"""
$ %prog ROOT

eg. 

$ %prog sprite.Sprite

def test_add(self):

    # Doc string for pygame.sprite.Sprite:

    ...
"""
)

################################### FUNCTIONS ##################################

def module_in_package(module, pkg):
    return ("%s." % pkg.__name__) in module.__name__

def get_package_modules(pkg):
    modules = (getattr(pkg, x) for x in dir(pkg) if is_public(x))
    return [m for m in modules if ismodule(m) and module_in_package(m, pkg)]
                                                 # Don't want to pick up 
                                                 # string module for example
def py_comment(input_str):
    return '\n'.join (
        [('# ' + l) for l in input_str.split('\n')]
    )

def is_public(obj_name):
    return not obj_name.startswith(('__','_'))

def is_test(f):
    return f.__name__.startswith('test_')

def get_callables(obj, if_of = None):
    publics = (getattr(obj, x) for x in dir(obj) if is_public(x))
    callables = [x for x in publics if callable(x)]

    if if_of:
        callables = [x for x in callables if if_of(x)] # isclass, ismethod etc
    
    return set(callables)

def get_class_from_test_case(TC, default = ''):
    TC = TC.__name__
    if 'Type' in TC:
        return '.' + TC[:TC.index('Type')]
    else:
        return default

################################################################################

def test_stub(f, module, parent_class = None):
    test_name = 'test_%s' % f.__name__
    unit_name = '%s.' % module.__name__

    if parent_class:
        unit_name += '%s.' % parent_class

    unit_name += f.__name__

    stub = STUB_TEMPLATE.render (

        test_name = test_name,
        comments = py_comment(getdoc(f) or ''),
        unitname = unit_name,
    )

    return test_name, stub

def names_of(*args):
    return tuple(map(lambda o: o.__name__, args))

def module_stubs(module):
    stubs = {}

    classes = get_callables(module, isclass)
    functions = get_callables(module) - classes

    for class_ in classes:
        if ("%s.%s" % names_of(module, class_)).startswith(IGNORE):
            continue

        for method in get_callables(class_):
            stub = test_stub(method, module, class_.__name__ )
            stubs['%s.%s.%s' % names_of(module, class_, method) ] = stub

    for function in functions:
        stub = test_stub(function, module)
        stubs['%s.%s' % names_of(module, function) ] = stub
    
    return stubs

def package_stubs(package):
    stubs = dict()

    for module in get_package_modules(package):
        stubs.update(module_stubs(module))
    
    return stubs

def already_tested_in_module(module):
    already = []

    mod_name =  module.__name__
    
    test_name = "%s_test" % mod_name[7:]
    
    try: test_file = __import__(test_name)
    except ImportError: 
        # TODO: maybe notify?
        return
    
    classes = get_callables(test_file, isclass)
    test_cases = (t for t in classes if TestCase in t.__bases__)
    
    for class_ in test_cases:
        class_tested = get_class_from_test_case(class_, default = '')
    
        for test in get_callables(class_, is_test):
            fname = test.__name__[5:].split('__')[0]
            already.append("%s%s.%s" % (mod_name, class_tested, fname))
    
    return already

def already_tested(package):
    already = []

    for module in get_package_modules(package):
        already.append(already_tested_in_module(module))
    
    return already

def get_stubs(root):
    module_root = module_re.search(root)
    if module_root:
        module = getattr(pygame, module_root.group(1))
        stubs = module_stubs(module)
        tested = already_tested_in_module(module)
    else:
        stubs = package_stubs(pygame)
        tested = already_tested(pygame)
    
    return stubs, tested

if __name__ == "__main__":
    options, args = opt_parser.parse_args()

    if args:
        root = args[0]
        if not root.startswith('pygame.'):
            root = 'pygame.%s.' % root

        stubs, tested = get_stubs(root)

        for fname in sorted(s for s in stubs.iterkeys() if s not in tested):
            if fname.startswith( root ):
                test_name, stub = stubs[fname]

                if options.list:
                    print ('%13s: %s\n%13s: %s\n' %
                          ('Callable Name', fname, 'Test Name', test_name))
                else:
                    print stub
    else:
        opt_parser.print_help()

################################################################################