# ICLR-style paper

LaTeX source for the ΔKV paper, using the official ICLR conference style
(vendored from the [ICLR Master Template](https://github.com/ICLR/Master-Template),
retargeted to ICLR 2027).

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Or: `make` in this directory.

The compiled PDF is a double-blind ICLR submission: anonymous authors,
side ruler, running header *Under review as a conference paper at ICLR 2027*.
A prebuilt `main.pdf` is in this folder. Numbers in Table 1 are from
`python -m deltakv bench --seq 64 --rank 4 --scale 0.01`.
