$(document).ready(function () {


	var _comments = {};

	$.getJSON("docscomments.json", function (data) {
		//console.log(data);
		for(var i=0; i < data.length; i+=1) {
			var link = data[i]['link']
			if(typeof _comments[link] === "undefined") {
				_comments[link] = [];
			}
			_comments[link].push(data[i]);
		}


		var getComments = function (link) {
			return _comments[link];
		};

		var searchButtonHtml = [
			'<form action="http://www.google.com/codesearch" class="addcomment"><input type="hidden" value="file:\.py$ ', 
			'" name="q"><input type="submit" value="Search internet source code for examples of pygame.display.init "></form>'
		];

		var addCommentHtml = [
			'<form action="comment_new.php" class="addcomment"><input type="hidden" value="',
			'" name="link"><input type="submit" value="Add a Comment"></form>'
		];
		var showCommentHtml = [
			'<a href="#comment_',
			'" title="show comments" class="commentButton">Comments ',
			'</a><div id="comment_',
			'" class="hidden"></div>'
		];
		var commentPartHtml = [
			'<div class="commentPart"><div class="commentHeading">', '</div>',
			'<div class="commentContent">', '</div></div>',
		]

		$('dt.title').each(function (idx, el) {
			var link = $(el).attr('id');
			if (typeof link === "undefined") {
				return;
			}

			// Add "search internet for source code" buttons.
			var searchButton = $(searchButtonHtml[0] + link + searchButtonHtml[1]);
			$(el).next().append(searchButton);

			// Add show comments buttons.
			var comments = getComments(link);
			if(typeof comments === "undefined") {
				var addCommentButton = $(addCommentHtml[0] + link + addCommentHtml[1]);
				$(el).next().append(addCommentButton);
			} else {
				// show comment button.
				//console.log(link)
				var $showCommentButton = $([showCommentHtml[0], 
							  link.replace(/\./g, "_"), 
							showCommentHtml[1],
							comments.length,
							showCommentHtml[2],
							  link.replace(/\./g, "_"), 
							showCommentHtml[3]
							  ].join(""));
				$(el).next().append($showCommentButton);

				$showCommentButton.click(function () {
					//console.log('asdf')
					var $commentSection = $("#comment_" + link.replace(/\./g, "_"));
					$commentSection.removeClass("hidden");
					$.each(comments, function(idx) {
						//console.log(comments[idx])
						// date + user
						var userName = comments[idx]['user'];
						if (userName == null) {
							userName = 'Anonymous';
						}
						var commentHeading = comments[idx]['datetimeon'] + " - " + userName;
						var commentContent= comments[idx]['content'];

						var $commentPart = $([commentPartHtml[0], 
							  commentHeading, 
							commentPartHtml[1],
							commentContent,
							commentPartHtml[2]].join("\n"));
						$commentSection.append($commentPart);
					});
					
				})
			}
			
		});



	});




});
