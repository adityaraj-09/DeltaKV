# arXiv-style preprint

Kour [arxiv-style](https://github.com/kourgeorge/arxiv-style): Times,
wide single column, “A Preprint” header, keywords, dated title block.

```bash
cd paper/arxiv
make
```

A compiled `main.pdf` is in this folder. Companion ICLR conference
formatting is in `paper/` (one directory up). Numbers match
`python -m deltakv bench --seq 64 --rank 4 --scale 0.01`.
