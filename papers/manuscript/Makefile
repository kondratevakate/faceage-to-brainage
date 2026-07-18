CC = latexmk
CFLAGS = -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make

.PHONY: all clean deep_clean

PDF = manuscript.pdf

all: $(PDF)

$(PDF): manuscript.tex references.bib
	$(CC) $(CFLAGS) manuscript.tex

clean:
	$(CC) -c
	rm -f *.nav *.snm

deep_clean: clean
	rm -f $(PDF)
