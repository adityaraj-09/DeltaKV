.PHONY: pdf clean

pdf: main.pdf

main.pdf: main.tex references.bib iclr2027_conference.sty iclr2027_conference.bst
	pdflatex -interaction=nonstopmode main
	bibtex main
	pdflatex -interaction=nonstopmode main
	pdflatex -interaction=nonstopmode main

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
