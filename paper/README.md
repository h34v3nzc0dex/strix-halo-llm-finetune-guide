# arXiv paper draft

Source for a workshop / arXiv-grade write-up of the two findings documented elsewhere in this repo:

1. The **ROCm/HIP vs Mesa RADV Vulkan precision inversion** on gfx1151 — Vulkan wins quantized decode by ~22%, ROCm wins BF16 decode by ~117%, driven by `bf16: 0` on the RADV STRIX_HALO codepath.
2. The **`GGML_HIP_ROCWMMA_FATTN=OFF`** finding — disabling the flag (against AMD's published RDNA 3.5 best-practices doc) yields up to 145% higher prefill at 8K context.

## Files

| File | Purpose |
|---|---|
| `paper.tex` | LaTeX source (compiles to ~7 pages with `pdflatex`) |
| `paper.pdf` | Rendered PDF (regenerate via `latexmk -pdf paper.tex`) |
| `make_figures.py` | Parses the raw bench logs from `../vulkan-vs-rocm-sweep/` and `../rocwmma-fattn-sweep/` into the three figures used in the paper |
| `figures/` | The three PDF figures (regenerate via `python3 make_figures.py`) |

## Compile

```bash
# install requirements once (Debian/Ubuntu)
sudo apt install -y texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended latexmk python3-matplotlib

# regenerate figures
python3 make_figures.py

# compile (two passes for cross-refs and bibliography)
latexmk -pdf paper.tex
```

## arXiv submission notes

- **Category:** `cs.LG` primary, `cs.PF` (performance) secondary
- **Authorship:** single author (Paul Durkin, ORCID 0009-0000-2537-1578). Per ICML/NeurIPS/Nature/IEEE/ACM policy, LLMs cannot be listed as authors; Claude (Anthropic) is acknowledged in the Acknowledgments section with explicit description of contribution.
- **Supplementary material:** raw bench logs are at the Hugging Face dataset `NorthstarAurora/strix-halo-bench-data` (cite via the BibTeX block included in that dataset's README)
- **License (paper):** CC BY 4.0 recommended (allows reuse with attribution)
- **License (code/data):** MIT (already set on the guide repo + bench dataset)

To upload:
1. Submit `paper.tex`, `paper.bbl` (after running pdflatex), and `figures/*.pdf` as a tarball to arxiv.org
2. ORCID linking happens automatically if the arXiv account is connected to ORCID
3. arXiv assigns an ID + DOI within a few hours of acceptance

## Possible workshop venues (in order of fit)

| Venue | Deadline cycle | Fit |
|---|---|---|
| **ML for Systems @ NeurIPS** | Sept | Systems-flavored ML, lower bar than main track |
| **MLSys** | Oct | Main systems-ML conference; harder bar but fits perfectly |
| **HotOS / OSDI workshop tracks** | varies | Systems angle could land here too |
| **USENIX ;login: magazine** | rolling | Practitioner-focused, rolling submission, lighter review |
