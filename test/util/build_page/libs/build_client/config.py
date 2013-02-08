################################################################################
# Imports

# StdLib
import os 
import sys
import cgi
import re
import glob
import ConfigParser
import pprint
import webbrowser

# User Lib
import callproc
from helpers import *

################################################################################

defaults = dict (

    ########################################################################
    # Paths

    src_path = normp('./pygame/trunk'),

    ########################################################################
    # Commands
    #

    config_cmd = [sys.executable, "config.py"],

    config_py_interaction = 'Y\nY\nY\n',

    build_cmd =  [sys.executable, "setup.py", "build"],
        
    # install_cmd is extrapolated below in configure function

    tests_cmd = [sys.executable, "run_tests.py"],
    
    ########################################################################
    # Environments

    build_env   = os.environ.copy(),

    ########################################################################
    # Make

    make_package = None,
)

################################################################################

class config_obj(object):
    def __init__(self, init=None):
        if init: self.__dict__.update(init)
    def __str__(self):
        return pprint.pformat(self.__dict__)

    __repr__ = __str__

    def htmlDump(self, open_in_browser = False):
        html = (
            "<hr><h3>%s</h3>"
            "<pre>%s</pre><hr>" % (
                self.platform_id,
                cgi.escape(str(self)).replace('\n', '<br />')
        ))

        if open_in_browser:
            config_html_file = '%s_config.html' % self.platform_id
            config_html = file(config_html_file, 'w')
            config_html.write (html)
            config_html.close()
            webbrowser.open(config_html_file)

        return html

def merge_dict(dest, indict):
    for key, val in indict.items():
        if (key in dest and isinstance(dest[key], dict) and
                            isinstance(val, dict)):
            merge_dict(dest[key], val)
        else:   
            dest[key] = val

def config_to_dict(config_file):
    config_data = ConfigParser.SafeConfigParser()
    config_data.read([config_file])

    defaults = set(config_data.items('DEFAULT'))
    config = {'DEFAULT' : dict(defaults)}

    for sections in config_data.sections():
        config[sections] = dict(set(config_data.items(sections)) - defaults)

    return config

################################################################################

def update_cmds_with_alternate_python(config):
    python = config['DEFAULT'].get('python_path')
    if python:
        for key in config:
            if key.endswith('_cmd'): config[key][0] = python

def merge_defaults_and_objectify_config(config_file):
    config = defaults.copy()
    config_dict = config_to_dict(config_file)

    merge_dict(config, config_dict)
    config.update(config_dict['DEFAULT'])

    update_cmds_with_alternate_python(config)
    
    return  config_obj(config)

################################################################################

def get_and_brand_latest_svn(src_path):
    if not os.path.exists(src_path): os.makedirs(src_path)
    
    rc, output = callproc.ExecuteAssertSuccess (
        ["svn","co","svn://seul.org/svn/pygame/trunk", src_path] )
    
    rev_match = re.search(r"(At)|(Checked out) revision ([0-9]+)\.", output)
    latest_rev = rev_match.group(3)

    callproc.ExecuteAssertSuccess(["svn","revert",src_path,"-R"])

    version_source = normp(src_path, 'lib/version.py')

    re_sub_file( version_source, 
                 r"(ver\s*=\s*)'([0-9]+\.[0-9]+\.[0-9]+[^']*)'", 
                 r"\1'\2-svn"+latest_rev+"'")

    return int(latest_rev)

def get_platform_and_previous_rev(config, config_file):
    config.platform_id = re.search (
        r"build_([^.]+).ini", os.path.basename(config_file) ).group(1)
    
    config.last_rev_filename = normp(
        "./last_rev_%s.txt" % config.platform_id )
    try:
        config.previous_rev = int(file(config.last_rev_filename, "r").read())
    except:
        config.previous_rev = 0

################################################################################

def extra_flags(flags):
    return [c for c in flags.split() if c]

def configure(config_file):
    # READ INI FILE
    c = merge_defaults_and_objectify_config(config_file)    
    
    # SUBVERSION
    get_platform_and_previous_rev(c, config_file)
    
    # Possibly updated between builds?? Here's a good spot?
    c.latest_rev = get_and_brand_latest_svn(c.src_path)
    
    # WORKING DIR 
    c.working_dir = normp(os.path.dirname(__file__))
    os.chdir(c.working_dir)

    # CONFIG.PY INTERACTION
    # ini files are parsed as raw strings
    c.config_py_interaction = c.config_py_interaction.replace('\\n','\n')

    # BUILD
    c.build_cmd += extra_flags(c.extra_build_flags)
    c.install_cmd = c.build_cmd[:]

    # INSTALLER
    if c.make_package:
        c.dist_path = os.path.join(c.src_path, 'dist')
        c.build_cmd += [c.make_package]

    # INSTALL / TEST CMDS / PATH
    c.temp_install_path = os.path.join(os.path.dirname(__file__),"install_test")
    c.temp_install_pythonpath = os.path.join (
        c.temp_install_path, c.test_dir_subpath
    )
    c.install_cmd += ['install', '--prefix', c.temp_install_path]
    c.tests_cmd += extra_flags(c.extra_test_flags)

    # INSTALL / TEST ENV
    test_env = {"PYTHONPATH" : c.temp_install_pythonpath}
    c.install_env = c.build_env.copy()  #TODO rename to test_install_env
    c.install_env.update(test_env)

    # RESULTS
    c.prebuilts_filename = normp("./output/prebuilt")
    c.buildresults_filename = normp("./output/buildresults")
    c.buildresults_zip = normp("./output/build.zip")

    # FILES TO ADD TO RESULTS ZIP
    c.buildresult_files = [
       normp('config', 'build_%s.ini' % c.platform_id),
       normp(c.src_path, 'Setup'),
    ]

    return c

################################################################################

def get_configs(args):
    search = "./config/build_%s.ini" % (args and args[0] or '*')
    config_file_list = glob.glob(search)

    for config_file in config_file_list:        
        yield configure(config_file)

def main():
    for conf in get_configs(sys.argv[1:]):
        conf.htmlDump(open_in_browser = True)

################################################################################

if __name__ == '__main__':
    main()
    
################################################################################