# Makefile for Project.tex
# Requires: latexmk, pdflatex, bibtex (TeX Live or MiKTeX)
#
# Usage:
#   make          – build Project.pdf
#   make clean    – remove auxiliary files
#   make distclean – remove auxiliary files AND the PDF

DOCNAME  := Project
LATEXMK  := latexmk
BUILDDIR := build.nosync
LMKFLAGS := -pdf -bibtex -interaction=nonstopmode -halt-on-error -outdir=$(BUILDDIR)

SECTIONS := $(wildcard sections/*.tex)
SOURCES  := $(DOCNAME).tex $(SECTIONS) references.bib

.PHONY: all clean distclean

all: $(DOCNAME).pdf

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(DOCNAME).pdf: $(SOURCES) | $(BUILDDIR)
	$(LATEXMK) $(LMKFLAGS) $(DOCNAME).tex
	cp $(BUILDDIR)/$(DOCNAME).pdf .

clean:
	rm -rf $(BUILDDIR)
	rm -f $(DOCNAME).bbl $(DOCNAME).blg $(DOCNAME).log $(DOCNAME).fdb_latexmk $(DOCNAME).fls

distclean: clean
	rm -f $(DOCNAME).pdf
