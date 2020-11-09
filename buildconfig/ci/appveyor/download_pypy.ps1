# For downloading pypy or pypy3.
#   powershell appveyor\\download_pypy.ps1 -pypy_version pypy2-v5.10.0-win32
#
# TODO: a better powershell dev would:
#         - make this function reusable.
#         - add sha or even md5 checksum verification.
function DownloadPyPy($which_pypy) {
    $webclient = New-Object System.Net.WebClient

    $which_pypy_zip = $which_pypy + ".zip"
    $download_url = "https://downloads.python.org/pypy/" + $which_pypy + ".zip"

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


function main ($pypy_version) {
   DownloadPyPy "pypy2.7-v7.3.2-win32"
   & DownloadPyPy "pypy3.6-v7.3.2-win32"
}

main $pypy_version
