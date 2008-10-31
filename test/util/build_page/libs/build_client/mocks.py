################################################################################
# Imports

# StdLib
import sys
import optparse
import os

# User Libs
import regexes 
import config
import update

from update_test import fixture, mock_config
from helpers import normp

################################################################################

svn_blame = lambda x: (0, '\n'.join([regexes.SVN_BLAME_TEST.strip()] * 1500))

get_and_brand_latest_svn = lambda x: 0

old_get_platform = config.get_platform_and_previous_rev
def get_platform_and_previous_rev(c, cfile):
    old_get_platform(c, cfile)
    c.previous_rev = -1

################################################################################

def skip_svn():
    update.svn_blame = svn_blame
    config.get_and_brand_latest_svn = get_and_brand_latest_svn

def force_build():    
    config.get_platform_and_previous_rev = get_platform_and_previous_rev

def run_tests():
    mock_config (
        tests_cmd        =    ['run_tests.py', 'sprite', '-s', '-F', 
                               '../../output/arstdhneoi.txt'],
        src_path         =    'pygame/trunk',
        install_env      =    os.environ.copy(),
    )

    update.run_tests()
    
    print update.run_tests.output

################################################################################

opt_parser = optparse.OptionParser()

opt_parser.add_option('-c', '--config',      action = 'store_true')
opt_parser.add_option('-s', '--skip_svn',    action = 'store_true')
opt_parser.add_option('-f', '--force_build', action = 'store_true')
opt_parser.add_option('-t', '--test',  action = 'store_true')

if __name__ == '__main__':
    options, args = opt_parser.parse_args()
    sys.argv = sys.argv[:1] + args
    
    if options.skip_svn:     skip_svn()
    if options.force_build:  force_build()
    
    if options.test:
        run_tests()
    else:
        # Swap out some slow components for quick debugging
        if options.config: config.main()
        else:              update.main()
    
################################################################################