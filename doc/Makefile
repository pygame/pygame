# You can set these variables from the command line.
PYTHON        = python
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d sphinx/build/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

.PHONY: rst help clean html web pickle htmlhelp latex changes linkcheck

all: help

rst: clean
	@$(PYTHON) create_rstref.py
	@cp -f BuildDarwin.txt builddarwin.rst
	@cp -f BuildMinGW.txt buildmingw.rst
	@cp -f BuildVC.txt buildvc.rst
	@cp -f BuildUnix.txt buildunix.rst
	@cp -f MODULES.txt newmodules.rst
	@cp -f MODULE_FAQ.txt module_faq.rst
	@cp -f RELEASE.txt release.rst
	@cp -f ../NEWS.txt news.rst

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      to make standalone HTML files"
	@echo "  pickle    to make pickle files"
	@echo "  json      to make JSON files"
	@echo "  htmlhelp  to make HTML files and a HTML help project"
	@echo "  latex     to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  changes   to make an overview over all changed/added/deprecated items"
	@echo "  linkcheck to check all external links for integrity"

docclean:
	@rm -rf sphinx/build *~ *.pyc *.orig ref/pygame2_*
	@rm -f ref/pygame2.rst ref/*~ ref/*.pyc ref/*.orig
	@rm -f builddarwin.rst buildmingw.rst buildunix.rst buildvc.rst
	@rm -f newmodules.rst module_faq.rst news.rst release.rst

clean: docclean
	@rm -rf html

html: rst
	mkdir -p sphinx/build/html sphinx/build/doctrees
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) sphinx/build/html
	@echo
	@echo "Build finished. The HTML pages are in sphinx/build/html."

pickle: rst
	mkdir -p sphinx/build/pickle sphinx/build/doctrees
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) sphinx/build/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

web: pickle

json: rst
	mkdir -p sphinx/build/json sphinx/build/doctrees
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) sphinx/build/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp: rst
	mkdir -p sphinx/build/htmlhelp sphinx/build/doctrees
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) sphinx/build/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in sphinx/build/htmlhelp."

latex: rst
	mkdir -p sphinx/build/latex sphinx/build/doctrees
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) sphinx/build/latex
	@echo
	@echo "Build finished; the LaTeX files are in sphinx/build/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
	      "run these through (pdf)latex."

changes: rst
	mkdir -p sphinx/build/changes sphinx/build/doctrees
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) sphinx/build/changes
	@echo
	@echo "The overview file is in sphinx/build/changes."

linkcheck: rst
	mkdir -p sphinx/build/linkcheck sphinx/build/doctrees
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) sphinx/build/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in sphinx/build/linkcheck/output.txt."
