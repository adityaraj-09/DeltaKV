.PHONY: pdf clean

pdf: main.pdf

main.pdf: main.tex references.bib arxiv.sty
	pdflatex -interaction=nonstopmode main
	bibtex main
	pdflatex -interaction=nonstopmode main
	pdflatex -interaction=nonstopmode main

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
