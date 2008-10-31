<?php
$build_tag = $_GET['tag'];
$results_file = "buildresults_".$build_tag.".txt";
$result_contents = explode("\n", file_get_contents($results_file));

?>

<html>

 <head>
  <title>Alert! Warnings for <? echo $build_tag; ?> Automated Pygame Build Page</title>
 </head>

<body>

 <h2>Alert! Warnings for <? echo $build_tag; ?> Automated Pygame Build Page</h2>
 <hr>
 <p>Below are the <a href="results/<?php echo "$build_tag/$result_contents[0]/"?>setup">build</a> warnings for the <? echo $build_tag; ?> build from the <a href="index.php">Spectacularly Adequate Automated Pygame Build Page</a>. Although most warnings are related to casting or types and are harmless, they may indicate a bug or error in the build.</p> 
 <hr>

 <table width=100% style='border-top:1px solid #006c03; background:#DDDD44;'>
  <tr>
   <td>Automated build warnings:</td>
  </tr>
 </table>
 <table width=100% border=1>
  <tr>
   <td><b><? echo $build_tag; ?></b></td>
  </tr>
  <tr>
   <td><p><? echo "Revision: " . $result_contents[0] . " Built: " . $result_contents[1]; ?></td>
  </tr>
  <tr>
   <td><p><? echo $result_contents[4]; ?></td>
  </tr>
 </table>
 
</body>

</html>