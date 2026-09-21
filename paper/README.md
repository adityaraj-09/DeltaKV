# Papers

| Layout | Path | Look |
|---|---|---|
| ICLR conference | [`paper/`](.) | Double-blind, side ruler, 5.5in column |
| Camera-ready article | [`paper/arxiv/`](arxiv/) | Named author, no preprint banner, 6.5in column |

```bash
cd paper && make          # ICLR
cd paper/arxiv && make    # arXiv
```

Numbers in both PDFs are from `python -m deltakv bench --seq 64 --rank 4 --scale 0.01`.

