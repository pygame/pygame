import os
import callproc
import ConfigParser

def scp(local_path, remote_file = None):
    if remote_file == None:
        remote_file = os.path.split(local_path)[1]
    config_file = "./config/upload.ini"
    config_data = ConfigParser.SafeConfigParser()
    config_data.read([config_file])

    file_vars = {"local_path":local_path, "remote_file":remote_file}
    command = config_data.get("DEFAULT", "scp", vars = file_vars)
    callproc.ExecuteAssertSuccess(command)
    