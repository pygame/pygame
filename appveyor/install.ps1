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


function main () {
    InstallPackage $env:PYTHON wheel
    & DownloadPrebuilt
}

main
