import sys
import os

from importlib.machinery import ExtensionFileLoader

def unbulk_dyn_load(name):
    foo = ExtensionFileLoader("pygame." + name, name + ".pyd").load_module()
    sys.modules["pygame." + name] = foo
    return foo

def unbulk_dyn_load_package_name(module_name, package_name, extension_name):
    foo = ExtensionFileLoader(package_name, extension_name).load_module()
    sys.modules[module_name] = foo
    return foo 