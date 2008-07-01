import os
import sys
import re
import callproc
import time
import glob
import ConfigParser
import shutil
import upload_results

def write_file_lines(filename, line_list):
    file_obj = file(filename, "w")
    for line in line_list:
        if not isinstance(line, str):
            line = str(line)
        file_obj.write(line)
        file_obj.write("\n")
    file_obj.close()
    
def re_sub_file(file_path, match, replace):
    content = file(file_path, "r").read()
    content, count = re.subn(match, replace, content)
    assert(count > 0)
    output = file(file_path, "w")
    output.write(content)
    output.close()

def assert_path_exists(path, description):
    if not os.path.exists(path):
        raise Exception("ERROR: can't find "+description+" at : "+path)

def cleardir(path_to_clear):
    for root, dirs, files in os.walk(path_to_clear, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

def GetAndBrandLatestFromSVN(src_path):
    output = callproc.ExecuteAssertSuccess(["svn","co","svn://seul.org/svn/pygame/trunk",src_path])
    
    rev_match = re.search(r"(At)|(Checked out) revision ([0-9]+)\.", output)
    latest_rev = int(rev_match.group(3))
    
    callproc.ExecuteAssertSuccess(["svn","revert",src_path,"-R"])
    
    version_source = src_path + '/lib/version.py'
    re_sub_file(version_source, r"(ver\s*=\s*)'([0-9]+\.[0-9]+\.[0-9]+[^']*)'", r"\1'\2-svn"+str(latest_rev)+"'")
    
    return latest_rev        

def AppendBlameInfoToErrorsByFile(src_root, errors_by_file, line_func = lambda(x) : int(x[0])):
    for error_file in errors_by_file:
        print "blame for",error_file
        ret_code, blame_output = callproc.GetReturnCodeAndOutput(["svn", "blame", error_file], src_root, lineprintdiv=100)
        if ret_code == 0:
            blame_lines = blame_output.split('\n')
            for error in errors_by_file[error_file]:
                line = line_func(error)
                line_match = re.match(r"\s*([0-9]+)\s+([^ ]+)\s([^\r\n]*)", blame_lines[line - 1])
                rev = line_match.group(1)
                user = line_match.group(2)
                line = line_match.group(3)
                error.append(user)
                error.append(line)
                error.append(rev)

def GetBuildWarningsHTML(src_path, build_output):
    warnings_by_file = {}
    warning_matches = re.findall(r"^([^\(\s]+\.c)(?:\(|:)([0-9]+)(?:\)|:) ?:? warning:? ([^\r\n]+)[\r\n]", build_output, re.MULTILINE)
    if len(warning_matches) > 0:
        print "WARNING - found",len(warning_matches),"warnings"
        for warning_match in warning_matches:
            warning_file, line, message = warning_match
            if warning_file not in warnings_by_file:
                warnings_by_file[warning_file] = []
            warnings_by_file[warning_file].append([line, message])

        AppendBlameInfoToErrorsByFile(src_path, warnings_by_file)
                
        web_friendly_warnings = []
        for warning_file in warnings_by_file:
            for warning in warnings_by_file[warning_file]:
                file_location = os.path.split(warning_file)[1] + ":" + warning[0] + " last rev: " + warning[-1] + ":" + warning[-3]
                code_line = warning[-2].replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")
                web_friendly_warnings.append(file_location + "<br>warning:" + warning[1] + '<br><code>' + code_line + '</code>')                
        return "<hr>".join(web_friendly_warnings)
    else:
        print "no warnings found in:"
        print build_output
        return ""
    
script_path = os.path.split(sys.argv[0])[0]
print 'executing pygamebuilder from:',script_path
if script_path != "": 
    os.chdir(script_path)
print "-------------------------"

if not os.path.exists("./source"):
    os.mkdir("./source")
src_path = './source/pygame'
latest_rev = GetAndBrandLatestFromSVN(src_path)

if not os.path.exists("./output"):
    os.mkdir("./output")
    
if len(sys.argv) > 1:
    config_file_list = "./config/build_"+sys.argv[1:]+".ini"
else:
    config_file_list = glob.glob("./config/build_*.ini")

for config_file in config_file_list:
  
    config_data = ConfigParser.SafeConfigParser()
    config_data.read([config_file])
    platform_id = os.path.split(config_file)[1].replace(".ini", "").replace("build_", "")

    last_rev_filename = "./output/last_rev_"+platform_id+".txt"

    assert(config_data.has_option("DEFAULT", "python_path"))
    python_path = config_data.get("DEFAULT", "python_path")
    assert_path_exists(python_path, "expected python version")

    print "-------------------------"
    print "building",platform_id,"with python at",python_path
    try:
        previous_rev = int(file(last_rev_filename, "r").read())
    except:
        print "WARNING: could not find last rev built"
        previous_rev = 0
    
    if latest_rev == previous_rev:
        print "exiting - already built rev",latest_rev
    else:
        print "building",latest_rev,"(last built %d)" % previous_rev
        valid_build_attempt = True
        
        build_env = {}
        for option in config_data.options("build_env"):
            build_env[option] = config_data.get("build_env", option)
            
        ret_code, output = callproc.InteractiveGetReturnCodeAndOutput([python_path, "config.py"], "Y\nY\nY\n", src_path, build_env)
        print output
        if ret_code != 0:
            print "ERROR running config.py!"
            assert(ret_code == 0)
    
        dist_path = src_path + "/dist"
        if os.path.exists(dist_path):
            cleardir(dist_path)

        package_command = config_data.get("DEFAULT", "make_package")
        ret_code, output = callproc.GetReturnCodeAndOutput([python_path, "setup.py", package_command], src_path, build_env)
        if ret_code == 0:
            build_warnings = GetBuildWarningsHTML(src_path, output)
            
            package_mask = config_data.get("DEFAULT", "package_mask")
            installer_dist_path = glob.glob(dist_path+"/"+package_mask)[0]
            print "got installer at:", installer_dist_path
            installer_filename = os.path.split(installer_dist_path)[1]
            installer_path = "./output/"+installer_filename
            shutil.move(installer_dist_path, installer_path)
            
            temp_install_path = os.path.join(os.getcwd(), "install_test")
            if os.path.exists(temp_install_path):
                cleardir(temp_install_path)
            else:
                os.mkdir(temp_install_path)

            test_subpath = config_data.get("DEFAULT", "test_dir_subpath")
            temp_install_pythonpath = os.path.join(temp_install_path, test_subpath)
            os.makedirs(temp_install_pythonpath)
            
            test_env = {"PYTHONPATH":temp_install_pythonpath}
            install_env = build_env.copy()
            install_env.update(test_env)

            print "installing to:",temp_install_path
            callproc.ExecuteAssertSuccess([python_path, "setup.py", "install", "--prefix", temp_install_path], src_path, install_env)
        
            print "running tests..."
            ret_code, output = callproc.GetReturnCodeAndOutput([python_path, "run_tests.py"], src_path, test_env)
            error_match = re.search("FAILED \([^\)]+=([0-9]+)\)", output)
            if ret_code != 0 or error_match != None:
                errors_by_file = {}
                error_matches = error_matches = re.findall(r"^((?:ERROR|FAIL): [^\n]+)\n+-+\n+((?:[^\n]+\n)+)\n", output, re.MULTILINE)
                if len(error_matches) > 0:
                    print "TESTS FAILED - found",len(error_matches),"errors"
                    for error_match in error_matches:
                        message, traceback = error_match
                        trace_top_match = re.search(r'File "([^"]+)", line ([0-9]+)', traceback)
                        error_file, line = trace_top_match.groups()
                        if error_file not in errors_by_file:
                            errors_by_file[error_file] = []
                        errors_by_file[error_file].append([line, message, traceback])
                    AppendBlameInfoToErrorsByFile(src_path, errors_by_file)
                    
                    for error_file in errors_by_file:
                        print "test failures in:", error_file
                        for error in errors_by_file[error_file]:
                            print error
                            
                    build_result = "Build Successful, Tests FAILED"                            
                    web_friendly_errors = []
                    for error_file in errors_by_file:
                        for error in errors_by_file[error_file]:
                            file_location = os.path.split(error_file)[1] + ":" + error[0] + " last rev: " + error[-1] + ":" + error[-3] 
                            web_friendly_errors.append(file_location + "<br>" + error[1])                
                    build_errors = "<hr>".join(web_friendly_errors)
                else:
                    build_result = "Build Successful, Invalid Test Results"
                    build_errors = output.replace("\n", "<br>")                            
                    print "ERROR - tests failed! could not parse output:"
                    print output
            else:   
                print "success! uploading..."
                result_filename = "./output/prebuilt_%s.txt" % platform_id
                write_file_lines(result_filename, [str(latest_rev), time.strftime("%Y-%m-%d %H:%M"), "uploading"])
                upload_results.scp(result_filename)
                upload_results.scp(installer_path)
                write_file_lines(result_filename, [str(latest_rev), time.strftime("%Y-%m-%d %H:%M"), installer_filename])
                upload_results.scp(result_filename)
                build_result = "Build Successful, Tests Passed"                            
                tests_run = re.findall(r"^loading ([\r\n]+)$", output, re.MULTILINE)
                
                test_text = [test + " passed" for test in tests_run] 
                build_errors = "<br>".join(test_text)
        else:
            error_matches = re.findall(r"^([^\(\s]+\.c)(?:\(|:)([0-9]+)(?:\)|:) ?:? error:? ([^\r\n]+)[\r\n]", output, re.MULTILINE)
            if len(error_matches) > 0:
                print "FAILED - found",len(error_matches),"errors"
                errors_by_file = {}
                for error_match in error_matches:
                    error_file, line, message = error_match
                    if error_file not in errors_by_file:
                        errors_by_file[error_file] = []
                    errors_by_file[error_file].append([line, message])

                AppendBlameInfoToErrorsByFile(src_path, errors_by_file)
                        
                for error_file in errors_by_file:
                    print "errors in:",error_file
                    for error in errors_by_file[error_file]:
                        print error

                build_result = "Build FAILED, Tests not run"                            
                web_friendly_errors = []
                for error_file in errors_by_file:
                    for error in errors_by_file[error_file]:
                        file_location = os.path.split(error_file)[1] + ":" + error[0] + " last rev: " + error[-1] + ":" + error[-3] 
                        web_friendly_errors.append(file_location + "<br>ERROR:" + error[1])                
                build_errors = "<hr>".join(web_friendly_errors)
            else:

                link_error_matches = re.findall(r"^([^\(\s]+)\.obj : error ([^\r\n]+)[\r\n]", output, re.MULTILINE)
                if len(link_error_matches) > 0:
                    build_result = "Link FAILED, Tests not run"                           
                    print "FAILED - found",len(link_error_matches),"errors"
                    build_errors = ""
                    for error_match in link_error_matches:
                        source_name, message = error_match
                        build_errors += source_name + " : " + message + "<br>"
                        
                else:                
                    exception_match = re.search(r"^Traceback \(most recent call [a-z]+\):[\r\n]+(.+[^\r\n]+Error:[^\r\n]+)", output, re.MULTILINE | re.DOTALL)
                    if exception_match != None:
                        build_result = "Build FAILED, Tests not run"                            
                        build_errors = exception_match.group(1).replace("\n", "<br>")
                            
                    else:
                        build_result = "Build FAILED, Tests not run"                           
                        build_errors = ""
                        error_matches = re.findall(r"^error: ([^\r\n]+)", output, re.MULTILINE)
                        for error_match in error_matches:
                            build_errors += error_match + "<br>"
    
                        print "FAILED - unrecognized errors in:"
                        print output
                    
            build_warnings = GetBuildWarningsHTML(src_path, output)
                    
        if valid_build_attempt:            
            result_filename = "./output/buildresults_%s.txt" % platform_id
            write_file_lines(result_filename, [latest_rev, time.strftime("%Y-%m-%d %H:%M"), build_result, build_errors, build_warnings])
            upload_results.scp(result_filename)
            file(last_rev_filename, "w").write(str(latest_rev))
            print "COMPLETED build of",latest_rev
            print "-------------------------"
        else:
            print "FAILED build attempt of",latest_rev
    