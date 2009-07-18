import os, sys, datetime, textwrap, glob
from inspect import *
from optparse import OptionParser
import relative_indentation

for relpath in ('../../','../'):
    sys.path.insert(0, os.path.abspath(os.path.normpath(os.path.join \
        (os.path.dirname(__file__), relpath))))

try:
    import pygame2.test.pgunittest as unittest
except:
    import test.pgunittest as unittest

if sys.version_info < (2, 5, 0):
    ismemberdescriptor = isdatadescriptor
    isgetsetdescriptor = isdatadescriptor

date = datetime.datetime.now().date()
STUB_TEMPLATE = relative_indentation.Template ( '''
    def ${test_name}(self):

        # __doc__ (as of %s) for ${unitname}:

        ${comments}

        self.fail() ''' % date, 

        strip_common = 0, strip_excess = 0
)

def py_comment (input_str):
    #lines = []
    #for line in input_str.split ('\n\n'):
    #    if len (line) > 68:
    #        lines += textwrap.wrap (input_str, 68)
    #    else:
    #        lines += [line]
    #    lines += ['']
    #return '\n'.join ([('# ' + l) for l in lines]).rstrip ('\n# ')
    return '\n'.join ([('# ' + l) for l in input_str.split ('\n')]).rstrip \
           ('\n# ')

def do_import (name):
    components = name.split ('.')
    classed = False
    try:
        mod = __import__ (name)
    except ImportError:
        # Perhaps a class name?
        mod = __import__ (".".join (components[:-1]))
        classed = True

    if len (components) > 1:
        if classed:
            parts = components[1:-1]
        else:
            parts = components[1:]
        for comp in parts:
            mod = getattr (mod, comp)
    return mod

def init_optparser ():
    opt_parser = OptionParser()
    opt_parser.add_option (
        "-l",  "--list", action = 'store_true',
        help   = "list callable names not stubs" )
    opt_parser.add_option (
        "-t",  "--test_names", action = 'store_true',
        help   = "list test names not stubs" )

    opt_parser.add_option (
        "-d",  "--docs", action = 'store_true',
        help   = "get (more detailed) docs using makeref.py" )
    opt_parser.set_usage("""$ %prog ROOT\n\neg.\n\n$ %prog sprite.Sprite""")
    return opt_parser

def get_class_stubs (modname, cls):
    parts = dir (cls)
    stubs = {}
    
    for what in parts:
        # Skip private parts.
        if what.startswith ("_"):
            continue
        try:
            obj = cls.__dict__[what]
        except:
            continue # Skip invalid ones
        if isgetsetdescriptor (obj) or ismemberdescriptor (obj):
            key, val = get_attr_stubs (modname, cls, obj)
            stubs[key] = val
        elif ismethod (obj) or ismethoddescriptor (obj) or isfunction (obj):
            key, val = get_method_stubs (modname, cls, obj)
            stubs[key] = val
        else:
            pass
    return stubs

def get_func_stubs (modname, obj):
    test_name = 'todo_test_%s_%s' % (modname.replace(".", "_"), obj.__name__)
    unit_name = "%s.%s" % (modname, obj.__name__)
    stub = STUB_TEMPLATE.render (
        test_name = test_name,
        comments = py_comment ( "%s\n" % (getdoc(obj))),
        unitname = unit_name,
    )
    return unit_name, (test_name, stub)

def get_attr_stubs (modname, cls, obj):
    test_name = 'todo_test_%s_%s_%s' % \
        (modname.replace(".", "_"), cls.__name__, obj.__name__)
    if cls:
        unit_name = "%s.%s.%s" % (modname, cls.__name__, obj.__name__)
    else:
        unit_name = "%s.%s" % (modname, obj.__name__)
    stub = STUB_TEMPLATE.render (
        test_name = test_name,
        comments = py_comment ( "%s\n" % (getdoc(obj))),
        unitname = unit_name,
    )
    return unit_name, (test_name, stub)
            
def get_method_stubs (modname, cls, obj):
    if cls:
        unit_name = "%s.%s.%s" % (modname, cls.__name__, obj.__name__)
        test_name = 'todo_test_%s_%s_%s' % \
            (modname.replace(".", "_"), cls.__name__, obj.__name__)
    else:
        unit_name = "%s.%s" % (modname, obj.__name__)
        test_name = 'todo_test_%s_%s' % \
            (modname.replace(".", "_"),  obj.__name__)
    stub = STUB_TEMPLATE.render (
        test_name = test_name,
        comments = py_comment ( "%s\n" % (getdoc(obj))),
        unitname = unit_name,
    )
    return unit_name, (test_name, stub)

def get_stubs (root):
    module = do_import (root)
    parts = dir (module)
    stubs = {}
    
    for what in parts:
        if what.startswith ("_"):
            continue # Skip private ones.

        obj = module.__dict__[what]
        if ismodule (obj):
            continue # No module deep-diving
        if isclass (obj) and obj.__module__ in root:
            stubs.update (get_class_stubs (module.__name__, obj))
        elif (isfunction (obj) or isbuiltin (obj)) and \
            obj.__module__ in root:
            key, val = get_func_stubs (module.__name__, obj)
            stubs[key] = val
        elif (ismethod (obj) or ismethoddescriptor (obj)) and \
            obj.__module__ in root:
            key, val = get_method_stubs (module.__name__, None, obj)
            stubs[key] = val
    return stubs

def prep_method_name (methodname):
    separated = methodname.rfind('__')
    if separated != -1:
        methodname = methodname[:separated]

    if methodname.startswith ('test_'):
        return 'todo_%s' % methodname
    else:
        return methodname
    
def get_tested ():
    testfiles = glob.glob (os.path.join ("test", "*_test.py"))
    classes = []
    tested = []
    for f in testfiles:
        #try:
        mod = do_import (f[:-3].replace (os.path.sep, "."))
        #except ImportError:
        #    pass
        for what in dir (mod):
            obj = mod.__dict__[what]
            
            if isclass (obj) and unittest.TestCase in obj.__bases__:
                classes.append (obj)
    
    for cls in classes:
        for what in dir (cls):
            if what.startswith ("_"):
                continue
            mth = getattr (cls, what)
            if (ismethod (mth) or isfunction(mth)) and \
               (mth.__name__.startswith ('test_') or
                mth.__name__.startswith ('todo_')):
                tested.append (prep_method_name (mth.__name__))

    return tested

if __name__ == "__main__":
    opt_parser = init_optparser ()
    options, args = opt_parser.parse_args ()
    
    if not sys.argv[1:]:
        sys.exit (opt_parser.print_help ())

    # This is the only pygame2 specific code portion :-D.
    if len (args) > 0:
        root = args[0]
    else:
        root = 'pygame2'
    if not root.startswith('pygame2'):
        root = '%s.%s' % ('pygame2', root)

    stubs = get_stubs (root)
    tested = get_tested ()
    for fname in sorted (stubs.keys ()):
        if not fname.startswith(root):
            continue  # eg. module.Class
        test_name, stub = stubs[fname]
        if test_name in tested:
            continue
        if options.list:
            print ("%s," % fname)
        elif options.test_names:
            print (test_name)
        else:
            print (stub)
