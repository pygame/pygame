################################################################################
# Imports

# StdLib
import sys
import os
import re
import cgi
import time

# User Libs
import callproc
import config
import upload_results

from regexes import *
from helpers import *

################################################################################
# Results

BUILD_FAILED       = "Build FAILED, Tests not run"
BUILD_LINK_FAILED   = "Link FAILED, Tests not run"

TESTS_PASSED       = "Build Successful, Tests Passed"
TESTS_FAILED       = "Build Successful, Tests FAILED"
TESTS_INVALID      = "Build Successful, Invalid Test Results"

BUILD_SUCCESSFUL   = "Build Successful"

################################################################################
# Format Strings for errors

# Any errors with these keys will have the corresponding values html escaped
ESCAPE_KEYS = ('traceback', 'blame_line')

FILE_INFO = "%(error_file)s:%(line)s last rev: %(revision)s:%(user)s"

FORMATS = {

    TESTS_FAILED : FILE_INFO  + (
        '<br />%(test)s'
        '<br /><pre>%(traceback)s</pre>' ),

    BUILD_FAILED : FILE_INFO  + (
        '<br>ERROR: %(message)s' ),

    "BUILD_WARNINGS" : FILE_INFO + (
        '<br>warning:%(message)s'
        '<br><code>%(blame_line)s</code>'),
}

def errors_by_file_4_web(errors_by_file, format, cb = None, join='<hr>'):
    format_string = FORMATS[format]

    web_friendly = []

    for error_file, errors in errors_by_file.items():
        for error in errors:
            error.update({'error_file': os.path.basename(error_file)})
            if cb: cb(error)

            for k in ESCAPE_KEYS:
                if k in error: error[k] = cgi.escape(error[k])

            web_friendly.append(format_string % error)
    
    return join.join(web_friendly).replace('\n', '<br />')

################################################################################

def add_blame_to_errors_by_file( src_root, errors_by_file, line_func = None):
    if not line_func: line_func = lambda error: int(error['line'])
    
    for error_file, errors in errors_by_file.items():
        print "blame for %s" % error_file

        ret_code, blame_output = callproc.GetReturnCodeAndOutput (
            ["svn", "blame", error_file], src_root, lineprintdiv = 100 )

        if ret_code is 0:
            blame_lines = blame_output.split('\n')

            for error in errors:
                line = line_func(error)
                blame = SVN_BLAME_RE.search(blame_lines[line - 1])
                error.update(blame.groupdict())

################################################################################

def categorize_errors_by_file(errors, add_blame = 1):
    errors_by_file = {}
    
    for error in errors:
        error_file = error['file']
        if error_file not in errors_by_file: errors_by_file[error_file] = []
        errors_by_file[error_file].append(error)
    
    if add_blame:
        add_blame_to_errors_by_file( config.src_path, errors_by_file )
    
    return errors_by_file

################################################################################

def build_warnings_html(build_output):
    warnings = [w.groupdict() for w in BUILD_WARNINGS_RE.finditer(build_output)]
    
    if warnings:
        warnings_by_file = categorize_errors_by_file(warnings)
        return errors_by_file_4_web (warnings_by_file, "BUILD_WARNINGS")

    return ""

################################################################################

def test_errors(output):
    errors = []
    for error in (e.groupdict() for e in ERROR_MATCHES_RE.finditer(output)):
        error.update(TRACEBACK_RE.search(error['traceback']).groupdict())
        errors.append(error)
    return errors

def parse_test_results(ret_code, output):
    failed_test = TESTS_FAILED_RE.search(output)
    errors = test_errors(output)

    if failed_test and errors:
        errors_by_file = categorize_errors_by_file(errors)
        web_friendly = errors_by_file_4_web(errors_by_file, TESTS_FAILED)

        return TESTS_FAILED, web_friendly

    elif ( (failed_test and not errors) or
           (ret_code is not 0 and not failed_test) ):

        return TESTS_INVALID, output.replace("\n", "<br>")
    
    else:
        tests_run = re.findall(r"loading ([^\r\n]+)", output)
        test_text = [test + " passed" for test in tests_run]

        return TESTS_PASSED, "<br>".join(test_text)

################################################################################

def parse_build_results(ret_code, output):
    # SUCCESS
    if ret_code is 0: return BUILD_SUCCESSFUL, ''

    # ERRORS
    errors = [e.groupdict() for e in BUILD_ERRORS_RE.finditer(output)]
    if errors:
        errors_by_file = categorize_errors_by_file (errors)
        web_friendly = errors_by_file_4_web(errors_by_file, BUILD_FAILED)
        return BUILD_FAILED, web_friendly
    
    # LINK ERRORS
    link_errors = [
        "%(source_name)s:%(message)s<br>" % s.groupdict()
        for s in LINK_ERRORS_RE.finditer(output)
    ]
    if link_errors: return BUILD_LINK_FAILED, ''.join(link_errors)

    # EXCEPTIONS 
    exceptions = BUILD_TRACEBACK_RE.search(output)
    if exceptions:
        errors = exceptions.groupdict()['traceback'].replace("\n", "<br>")
        return BUILD_FAILED, errors
    
    # UNKNOWN ERRORS
    error_matches = re.findall(r"^error: ([^\r\n]+)", output, re.MULTILINE)
    if error_matches:
        return BUILD_FAILED, ''.join(["%s<br>" % m for m in error_matches])
    
    # ELSE
    return BUILD_FAILED, output.replace("\n", "<br>")

################################################################################

def configure_build():
    interaction = ''.join(['%s\n' % a for a in config.config_py_interaction])
    
    ret_code, output = callproc.InteractiveGetReturnCodeAndOutput (
        config.config_cmd, interaction, config.src_path, config.build_env
    )
    assert ret_code is 0

def build():
    return callproc.GetReturnCodeAndOutput (
        [c for c in config.build_cmd + [config.make_package] if c], 
        config.src_path, config.build_env
    )

def install():
    return callproc.ExecuteAssertSuccess (
        config.install_cmd + [config.temp_install_path], 
        config.src_path, config.install_env
    )

def run_tests():
    return callproc.GetReturnCodeAndOutput (
        config.tests_cmd, config.src_path, config.test_env
    )

################################################################################

def prepare_build_env():
    if config.make_package:
        if os.path.exists(config.dist_path): cleardir(config.dist_path)

    prepare_dir(config.temp_install_path)
    os.makedirs(config.temp_install_pythonpath)

################################################################################

def upload_build_results(build_result, build_errors, build_warnings):
    write_file_lines (
        config.buildresults_filename,
        ( [config.latest_rev, time.strftime("%Y-%m-%d %H:%M"),
           build_result, build_errors, build_warnings] )
    )

    upload_results.scp(config.buildresults_filename)
    file(config.last_rev_filename, "w").write(str(config.latest_rev))

def upload_installer():
    print "TODO"

################################################################################

def update_build():
    configure_build()
    ret_code, build_output = build()

    build_result, build_errors = parse_build_results(ret_code, build_output)
    build_warnings = build_warnings_html(build_output)

    if build_result is BUILD_SUCCESSFUL:
        install()
        build_result, build_errors = parse_test_results(*run_tests())

        if build_result is TESTS_PASSED:
            upload_installer()

    print '\n%s\n' % build_result

    upload_build_results(build_result, build_errors, build_warnings)

################################################################################

def main():
    # All configuration done before hand in config.py. + ini files Treat config
    # as readonly from here on in.

    global config
    for config in config.get_configs(sys.argv[1:]):
        if config.previous_rev < config.latest_rev:
            prepare_build_env()
            update_build()

if __name__ == '__main__':
    main()

################################################################################