<?php

mb_language('uni');
mb_internal_encoding('UTF-8');

$platform = $_GET['tag'];
$info = $_GET['info'];

$results_zip = "buildresults_".$platform.".zip";

function zip_index($zip)
{    
    $index = array();
    
    for ($i=0; $i < $zip->numFiles; $i++)
    {
        $stat = $zip->statIndex($i);
        $index[]=  $stat['name'];
    }
    
    sort($index);
    
    return $index;
}

function slug($str)
{
    $str = strtolower(trim($str));
    $str = preg_replace('/[^a-z0-9-]/', '-', $str);
    $str = preg_replace('/-+/', "-", $str);
    return $str;
}

function enumerate_zip_contents($results_zip, $platform, $info)
{
    $zip = new ZipArchive;
    
    if ($zip->open($results_zip) === TRUE)
    {
        
        $index = zip_index($zip);
        $content_file = $index[0];
        
        $tabs = array();
        
        foreach ($index as $i => $compressed_file)
        {   
            $slug = slug($compressed_file);
            
            if ($info == $slug)
            {   
                $content_file = $compressed_file;
            }
            
            $tabs[] = "<li class='t$i'><a class='t$i tab' href='?tag=$platform&info=$slug'> $compressed_file </a></li>";
        }

        $contents = htmlentities($zip->getFromName($content_file), ENT_QUOTES, 'UTF-8');
        $tab_contents = "<div class='t$i'><pre>$contents</pre></div>";
                
        $tabs = implode('', $tabs);
        
    } else die("Failed to open zip $results_zip");

    return <<<HTM
    <html>

    <head>
        <meta http-equiv="Content-Type" content="text/html" charset="utf-8" />

        <title> Build Info for $platform </title>    
    
        <link href="./results.css" rel="stylesheet" type="text/css"></link>
    
    </head>
    
    <body>
        
        <h1> <a href="$results_zip"> $results_zip </a> </h1> 
        
        <p><a href="warnings.php?tag=$platform"> warnings </a></p>
        <p><a href="builds.php"> builds </a></p>
        
        <div class="tabbed">
            
            <ul class="tabs"> 
                
                $tabs
            
            </ul>
            
                $tab_contents
                        
        </div>
    </body>
    </html>
HTM;

}

echo enumerate_zip_contents($results_zip, $platform, $info);

?>