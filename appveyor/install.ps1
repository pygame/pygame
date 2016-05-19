# Sample script to install Python and pip under Windows
# Authors: Olivier Grisel and Kyle Kastner
# License: CC0 1.0 Universal: http://creativecommons.org/publicdomain/zero/1.0/

function InstallPackage ($python_home, $pkg) {
    $pip_path = $python_home + "/Scripts/pip.exe"
    & $pip_path install $pkg
}

function DownloadPrebuilt ($url) {
    $webclient = New-Object System.Net.WebClient

    $filename = "prebuilt.zip"
    $basedir = $pwd.Path + "\"
    $filepath = $basedir + $filename
    if (Test-Path $filename) {
        Write-Host "Reusing" $filepath
        return $filepath
    }

    # Download and retry up to 5 times in case of network transient errors.
    Write-Host "Downloading" $filename "from" $url
    $retry_attempts = 3
    for($i=0; $i -lt $retry_attempts; $i++){
        try {
            $webclient.DownloadFile($url, $filepath)
            break
        }
        Catch [Exception]{
            Start-Sleep 1
        }
   }
   Write-Host "File saved at" $filepath
   return $filepath
}


function main () {
    InstallPackage $env:PYTHON wheel
    DownloadPrebuilt("https://bitbucket.org/pcraven/pygame/downloads/prebuilt.zip")
    & 7z x prebuilt.zip
    Rename-Item prebuilt prebuilt-x86
}

main
