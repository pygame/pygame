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

    src_path = './pygame/trunk',

    ########################################################################
    # Commands
    #

    config_cmd = [sys.executable, "config.py"],

    config_py_interaction = 'Y\nY\nY\n',

    build_cmd =  [ 
        sys.executable, "setup.py", "build", 
    ],
        
    install_cmd  = [
        sys.executable, "setup.py", "install", "--prefix", #temp_test_install
    ],

    tests_cmd = [sys.executable, "run_tests.py", 'event'],
    
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
    
    def htmlDump(self):
        return ("<hr><h3>%s</h3>"
                "<pre>%s</pre><hr>" % (
                    self.platform_id,
                    cgi.escape(str(self)).replace('\n', '<br />')
                ))

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

    config = dict ()
    for sections in config_data.sections() or ['DEFAULT']:
        config[sections] = dict(config_data.items(sections))

    return config

################################################################################

def update_cmds_with_alternate_python(config):
    python = config['build'].get('python_path')
    if python:
        for key in config:
            if key.endswith('_cmd'): config[key][0] = python

def merge_defaults_and_objectify_config(config_file):
    config = defaults.copy()
    config_dict = config_to_dict(config_file)

    merge_dict(config, config_dict)
    config.update(config_dict['build'])

    update_cmds_with_alternate_python(config)
    
    return  config_obj(config)

################################################################################

def get_and_brand_latest_svn(src_path):
    if not os.path.exists(src_path): os.makedirs(src_path)
    
    output = callproc.ExecuteAssertSuccess(
        ["svn","co","svn://seul.org/svn/pygame/trunk", src_path])
    
    rev_match = re.search(r"(At)|(Checked out) revision ([0-9]+)\.", output)
    latest_rev = int(rev_match.group(3))

    callproc.ExecuteAssertSuccess(["svn","revert",src_path,"-R"])
    
    version_source = src_path + '/lib/version.py'
    
    re_sub_file( version_source, 
                 r"(ver\s*=\s*)'([0-9]+\.[0-9]+\.[0-9]+[^']*)'", 
                 r"\1'\2-svn"+str(latest_rev)+"'")
    
    return latest_rev

def get_platform_and_previous_rev(config, config_file):
    config.platform_id = re.search (
        r"build_([^.]+).ini", os.path.basename(config_file) ).group(1)
    
    config.last_rev_filename = "./output/last_rev_%s.txt" % ( 
        config.platform_id )
    try:
        config.previous_rev = int(file(config.last_rev_filename, "r").read())
    except:
        config.previous_rev = 0

################################################################################

def configure(c):
    # SUBVERSION
    # Possibly updated between builds?? Here's a good spot?
    c.latest_rev = get_and_brand_latest_svn(c.src_path)

    # BUILD
    c.build_cmd += c.extra_build_flags.split(' ')

    # INSTALLER
    if c.make_package:
        c.dist_path = os.path.join(c.src_path, 'dist')
    
        c.installer_dist_path = glob.glob (
            os.path.join(c.dist_path, c.package_mask))[0]
    
        c.installer_filename = os.path.basename(c.installer_dist_path)
    
        c.installer_path = os.path.join('./output', c.installer_filename)

        c.build_cmd += [c.make_package]

    # INSTALL / TEST PATH
    c.temp_install_path = os.path.join(os.getcwd(), "install_test")
    c.temp_install_pythonpath = os.path.join (
        c.temp_install_path, c.test_dir_subpath
    )
    c.install_cmd += [c.temp_install_path]
    
    # INSTALL / TEST ENV
    c.test_env = {"PYTHONPATH" : c.temp_install_pythonpath}
    c.install_env = c.build_env.copy()
    c.install_env.update(c.test_env)

    # RESULTS
    c.prebuilts_filename = "./output/prebuilt_%s.txt" % c.platform_id
    c.buildresults_filename = "./output/buildresults_%s.txt" % c.platform_id

################################################################################

def get_configs(args):
    search = "./config/build_%s.ini" % (args and args[0] or '*')

    config_file_list = glob.glob(search)

    for config_file in config_file_list:
        config = merge_defaults_and_objectify_config(config_file)
        get_platform_and_previous_rev(config, config_file)
        configure(config)

        yield config

################################################################################

if __name__ == '__main__':
    for conf in get_configs(sys.argv[1:]):
        config_html_file = '%s_config.html' % conf.platform_id
        config_html = file(config_html_file, 'w')
        config_html.write (conf.htmlDump())
        config_html.close()
        webbrowser.open(config_html_file)

################################################################################