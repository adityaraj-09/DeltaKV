# Papers

| Layout | Path | Look |
|---|---|---|
| ICLR conference | [`paper/`](.) | Double-blind, side ruler, 5.5in column |
| arXiv preprint | [`paper/arxiv/`](arxiv/) | Named author, “A Preprint” header, 6.5in column |

```bash
cd paper && make          # ICLR
cd paper/arxiv && make    # arXiv
```

Numbers in both PDFs are from `python -m deltakv bench --seq 64 --rank 4 --scale 0.01`.

