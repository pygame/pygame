function InstallPackage ($python_home, $pkg) {
    $pip_path = $python_home + "/Scripts/pip.exe"
    & $pip_path install $pkg
}

function DownloadPrebuilt () {
    $webclient = New-Object System.Net.WebClient

    $download_url = "https://bitbucket.org/llindstrom/pygame/downloads/"
    $build_date = "20150922"
    $target = "x86"
    if ($env:PYTHON_ARCH -eq "64") {
        $target = "x64"
    }
    $prebuilt_file = "prebuilt-"+$target+"-pygame-1.9.2-"+$build_date+".zip"
    $prebuilt_url = $download_url + $prebuilt_file
    $prebuilt_zip = "prebuilt-" + $target + ".zip"

    $basedir = $pwd.Path + "\"
    $filepath = $basedir + $prebuilt_zip
    if (Test-Path $filepath) {
        Write-Host "Reusing" $filepath
        return $filepath
    }

    # Download and retry up to 5 times in case of network transient errors.
    Write-Host "Downloading" $filename "from" $prebuilt_url
    $retry_attempts = 3
    for($i=0; $i -lt $retry_attempts; $i++){
        try {
            $webclient.DownloadFile($prebuilt_url, $filepath)
            break
        }
        Catch [Exception]{
            Start-Sleep 1
        }
   }
   Write-Host "File saved at" $filepath

   & 7z x $prebuilt_zip
}

// TODO: a better powershell dev would make this function reusable.
//       add sha or even md5 checksum verification.
function DownloadPyPy($which_pypy) {
    $webclient = New-Object System.Net.WebClient

    $which_pypy_zip = $which_pypy + ".zip"
    $download_url = "https://bitbucket.org/pypy/pypy/downloads/" + $which_pypy + ".zip"

    $filepath = "$env:appveyor_build_folder\" + $which_pypy + ".zip"

    Write-Host "Downloading" $filepath "from" $download_url
    $retry_attempts = 3
    for($i=0; $i -lt $retry_attempts; $i++){
        try {
            $webclient.DownloadFile($download_url, $filepath)
            break
        }
        Catch [Exception]{
            Start-Sleep 1
        }
   }
   Write-Host "File saved at" $filepath

   & 7z x $which_pypy_zip
   $env:path = "$env:appveyor_build_folder\$which_pypy;$env:path"
}


function main () {
    DownloadPyPy "pypy2-v5.10.0-win32"
    & DownloadPyPy "pypy3-v5.10.1-win32"
    & InstallPackage $env:PYTHON wheel
    & DownloadPrebuilt
}

main
