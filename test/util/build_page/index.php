<?php
function sandwiched_file_list($prefix, $suffix)
{
	$localfilelist = scandir(".");
	$file_map = array();
	foreach ($localfilelist as $filename)
	{
		if (strlen($filename) > strlen($prefix) + strlen($suffix))
		{
			if (strncasecmp($filename, $prefix, strlen($prefix)) == 0)
			{
				$testsuffix = substr($filename, strlen($filename) - strlen($suffix));
				if (strcasecmp($testsuffix, $suffix) == 0)
				{
					$tag = substr($filename, strlen($prefix), strlen($filename) - strlen($prefix) - strlen($suffix));
					$contents = explode("\n", file_get_contents($filename));
					$file_map[$tag] = $contents;
				}
			}
		}
	}
	return $file_map;
}

$prebuilt_list = sandwiched_file_list("prebuilt_", ".txt");
$results_list = sandwiched_file_list("buildresults_", ".txt");

?>

<html>

 <head>
  <title>The Spectacularly Adequate Automated Pygame Build Page</title>
 </head>

<body>

 <h2>The Spectacularly Adequate Automated Pygame Build Page</h2>
 <hr>
 <p>These automated builds are Prerelease Unstable from the latest SVN sources - but all posted installers have passed unittests, so they should be suitable for evaluating new features or confirming that bugs have been fixed.</p> 
 <p>NOTE: these builds modify the pygame.version.ver string to include the SVN revision they were built from - please note the revision number when <a href='http://pygame.motherhamster.org/bugzilla/'>reporting bugs</a> with these builds.</p> 
 <hr>
 <table width=100% style='border-top:1px solid #00036c; background:#88CCFF;'>
  <tr>
   <td> Installers:</td>
  </tr>
 </table>
 <table width=100% border=1>
  <tr>
<?php
	foreach ($prebuilt_list as $platform_tag=>$platform_file)
	{
		echo "<td><b>" . $platform_tag . "</b></td>";
	}
 ?>
  </tr>
  <tr>
<?php
	foreach ($prebuilt_list as $platform_tag=>$platform_file)
	{
		echo "<td><p>Revision: " . $platform_file[0] . " Built: " . $platform_file[1] . "</td>";
	}
 ?>
  </tr>
  <tr>
<?php	
	foreach ($prebuilt_list as $platform_tag=>$platform_file)
	{
		if (strncasecmp($platform_file[2], "uploading", 9) == 0)
		{
			echo "<td><p>Uploading... please wait</p></td>";
		}
		else
		{    
            if (preg_match('/failed_tests_(.*)/', $platform_file[2], $group)) {
                $passing_installer = $group[1];
                
                echo "<td>Most Recent Build: <br /><a href='" . $platform_file[2] . "'>" . $platform_file[2] . "</a><br />";
                echo "<br />Most Recent Build Successfuly Passing Tests: <br/>(see pygame.version.ver for svn version number)<br/>";
                
                if (file_exists($passing_installer))
                {
                    echo "<a href='" . $passing_installer . "'>  " . $passing_installer . "</a></td>";
                }
                else
                {
                    echo "No passing build yet!";
                }
                
            } else {
                echo "<td><a href='" . $platform_file[2] . "'>" . $platform_file[2] . "</a></td>";
            }
		}
	}
 ?>
  </tr>
 </table>

 <br>

 <table width=100% style='border-top:1px solid #006c03; background:#88FFCC;'>
  <tr>
   <td>Most recent Automated build results:</td>
  </tr>
 </table>
 <table width=100% border=1>
  <tr>
<?php	
	foreach ($results_list as $platform_tag=>$platform_file)
	{
		echo "<td><b>" . $platform_tag . "</b></td>";
	}
 ?>
  </tr>
  <tr>
<?php	
	foreach ($results_list as $platform_tag=>$platform_file)
	{
		echo "<td><p>Revision: " . $platform_file[0] . " Built: " . $platform_file[1] . "</td>";
	}
 ?>
  </tr>
  <tr>
<?php	
	foreach ($results_list as $platform_tag=>$platform_file)
	{
		echo "<td><p>" . $platform_file[2] . "</p></td>";
	}
 ?>
  </tr>
  <tr>
<?php

	foreach ($results_list as $platform_tag=>$platform_file)
	{
		echo "<td>";
		if (trim($platform_file[3]) != "")
		{
			echo "<p><FONT COLOR='#FF0000'>" . $platform_file[3] . "</p>";
		}
		if (trim($platform_file[4]) != "")
		{
			echo "<a href='warnings.php?tag=$platform_tag'>view warnings </a>";
            echo "&nbsp;<a href='results/$platform_tag/$platform_file[0]/setup'> build info </a>";
		}
        echo "</td>";
        
	}    
 ?>
  </tr>
 </table>

</body>

</html>