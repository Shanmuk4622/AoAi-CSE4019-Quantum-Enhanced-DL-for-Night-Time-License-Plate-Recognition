# CENITH-T Paper → IEEE ACCESS Format Conversion

## Overview

Convert `cenith_t_ieee_double.tex` (currently in `IEEEtran` double-column journal format) into the **IEEE ACCESS** format (`ieeeaccess` class) while critically reviewing and improving the paper's content, structure, and LaTeX quality.

The output file will be: `cenith_t_access.tex` in the LAB directory, structured to work with the `ACCESS_latex_template_20240429/` directory's class and style files.

---

## Key Differences: IEEEtran → ieeeaccess

| Element | Current (IEEEtran) | Target (ieeeaccess) |
|---|---|---|
| `\documentclass` | `[journal,twocolumn]{IEEEtran}` | `{ieeeaccess}` |
| Title | `\title{...}` | `\title{...}` (same macro, different rendering) |
| Authors | Custom `\author{...}` with superscripts | `\author{\uppercase{...}\authorrefmark{N}}` per author |
| Affiliations | Inline in `\author{}` | `\address[N]{...}` for each author separately |
| Correspondence | None | `\corresp{Corresponding author: ...}` |
| Footer note | None | `\tfootnote{...}` for funding/support |
| Running head | None | `\markboth{...}{...}` |
| DOI/History | None | `\doi{...}` and `\history{...}` |
| Abstract | `\begin{abstract}` | Same, but 150–250 words, no abbreviations |
| Keywords | `\begin{IEEEkeywords}` | `\begin{keywords}` |
| `\titlepgskip` | Not needed | `-21pt` after keywords |
| `\maketitle` | After `\begin{document}` | After `\titlepgskip` |
| Section intros | No `\PARstart` | First section uses `\PARstart{F}{irst}` |
| Figure env | `\begin{figure*}` for wide | `\Figure[t!](topskip=0pt,...){img}{caption}` macro |
| Biography | `\begin{IEEEbiography}[photo]{Name}` | Same (works with ieeeaccess too) |
| `\EOD` | Not present | Required before `\end{document}` |
| Bibliography | `\begin{thebibliography}{}` | Same, but must use IEEE format citations |
| `\usepackage{booktabs}` | Yes | Remove — not in ACCESS template |
| `\usepackage{algorithm}` | Yes | Keep — used for Algorithm 1 |
| `\usepackage{hyperref}` | Yes | Remove — ieeeaccess has its own hyperref handling |

---

## Content Issues to Fix

### Problems Identified in Current Paper

1. **Literal `\\n` in source**: Lines 71, 114, 140, 157, 170, 384, 392 contain literal `\\n` escape sequences that were programmatically inserted (from `expand_ieee_more.py`). These are **not real LaTeX newlines** — they must be converted to proper `\n` paragraph breaks or logical section breaks.

2. **Garbled subsections**: Multiple subsections (`\subsection{Neuro-Symbolic AI}`, `\subsection{Graph Message Passing}`, `\subsection{Vector Space Retrieval}`, `\subsection{Theoretical Bounds}`, `\subsection{Computational Complexity}`, `\subsection{Cross-Domain Generalization}`, `\subsection{Hyperparameter Sensitivity}`) were concatenated into single long lines — they need to be properly separated.

3. **Missing \usepackage{algorithm}** pair — the `algorithmic` package is imported but `algorithm` package is also needed (currently it is there, keep it).

4. **Table formatting**: Tables use `|l|l|` borders with `\hline` — ACCESS format prefers cleaner `booktabs`-style or simple tables. However, since `booktabs` isn't in access template, we'll use the IEEE-standard pipe style but clean up the training config table.

5. **Abstract length**: Current abstract is ~180 words — within 150–250 limit, good. But it starts with abbreviations ("LLMs") without spelling out first — fix.

6. **Qualitative example**: The "Skeleton vs Skin" section mentions "Erik Axel Karlfeldt's association with the venue" for a physics Nobel Prize venue — this is factually incorrect (Karlfeldt was a literature Nobel laureate, not an architect). This looks like a hallucinated example. Should be cleaned up or generalized.

7. **Image paths**: Currently `Templates/Picture1.jpg` etc. Since new file will be in the same LAB directory, paths stay as `Templates/PictureN.jpg`. ✓

8. **Author photo paths**: Currently `Templates/Deepthi.jpg` and `Templates/Shanmukesh.png` — paths stay the same. ✓

9. **Bibliography**: Citations are in prose format — need to be converted to IEEE numbered format (`[1] J. K. Author, "Title,"...`).

---

## Proposed Changes

### [NEW] `cenith_t_access.tex` (LAB root directory)

Full rewrite of the document with:

#### Preamble
- `\documentclass{ieeeaccess}`
- Packages: `cite`, `amsmath`, `amssymb`, `amsfonts`, `algorithmic`, `graphicx`, `textcomp`, `algorithm`
- Bold math setup (from ACCESS template)
- `\BibTeX` definition

#### Header Section
- `\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}`
- `\doi{10.1109/ACCESS.2026.XXXXXXX}` (placeholder)
- `\title{CENITH-T: ...}` (unchanged)
- `\author{\uppercase{Deepthi Godavarthi}\authorrefmark{1} and \uppercase{Shanmukesh Bonala}\authorrefmark{2}}`
- `\address[1]{School of CSE, VIT-AP University, ...}` + email
- `\address[2]{School of CSE, VIT-AP University, ...}` + email
- `\tfootnote{This work was conducted as part of academic research at VIT-AP University...}`
- `\markboth{Godavarthi and Bonala: CENITH-T: Confidence-Enhanced Neural Inference...}{...}`
- `\corresp{Corresponding author: Shanmukesh Bonala (e-mail: Shanmukesh.23bce20070@vitapstudent.ac.in).}`

#### Abstract
- Fix: spell out "Large Language Models (LLMs)" before using abbreviation
- Keep ~180 words (within limit)
- Remove any abbreviations used without first defining them

#### Keywords
- Change to `\begin{keywords}` ... `\end{keywords}`
- Sort alphabetically (ACCESS requirement)

#### Body Sections
- Add `\titlepgskip=-21pt` and `\maketitle` after keywords
- Fix Introduction opening with `\PARstart{T}{he}` 
- Un-concatenate all the garbled sections (Literature Review subsections, Methodology subsections)
- Fix all literal `\\n` escape sequences
- Properly structure the HSG Graph Message Passing, Vector Space Retrieval, Theoretical Bounds, Computational Complexity, Cross-Domain, Hyperparameter Sensitivity as proper subsections

#### Figures
- Keep `\begin{figure*}` for Picture1.jpg (architecture figure, full-width)
- Keep standard `\begin{figure}` for smaller figures (Pictures 2-5)
- The `\Figure[]{}{}` macro from ACCESS is optional — standard `\begin{figure}` also works

#### Algorithm
- Keep `\begin{algorithm}[htbp]` with `\begin{algorithmic}[1]` — compatible

#### Tables
- Clean up Training Config table — use `tabular` with `|l|l|` (same style)
- Keep other tables with `booktabs` style... wait, remove `\usepackage{booktabs}` and convert `\toprule/\midrule/\bottomrule` to `\hline` in all tables since ieeeaccess doesn't include booktabs

#### Bibliography
- Keep as `\begin{thebibliography}{00}` 
- Convert all citations to proper IEEE numbered format (currently in author-year prose)
- Fix `\bibitem` keys to match `\cite{}` references in text

#### Biographies  
- Keep `\begin{IEEEbiography}` with photos — same syntax works in ieeeaccess
- Add `\EOD` before `\end{document}`

---

## Verification Plan

### Build Test
```
cd "d:\Documents\V-TOP\Winter-Sem 2025-26\NLP CSE3015\LAB"
pdflatex -interaction=nonstopmode cenith_t_access.tex
```
Run twice for cross-references. Check for errors, warnings. The output PDF should match the ACCESS journal style (cyan/teal color theme, IEEE Access logo header).

### Manual Verification
- Check PDF opens correctly and has the ACCESS journal header format
- Verify all figures appear
- Verify algorithm block renders
- Verify author photos render
- Confirm no `??` for unresolved references

---

## Open Questions

> [!IMPORTANT]
> **Qualitative Example Accuracy**: In the "Skeleton vs. Skin" analysis, the example mentions "Erik Axel Karlfeldt's association with the venue" for a 1964 Physics Nobel Prize venue. This appears to be a hallucinated fact (Karlfeldt was a Literature Nobel laureate, not an architect). Should this be:
> - (a) Removed entirely
> - (b) Replaced with a cleaner, non-specific example
> - (c) Left as-is (if this is intentionally a demonstration of the model's behavior)

> [!IMPORTANT]
> **Funding Acknowledgment**: The `\tfootnote{}` in the ACCESS format requires a funding/support statement. Should I use a generic statement like "This work was conducted as part of academic research at VIT-AP University, Amaravati, Andhra Pradesh, India" or do you have specific funding to acknowledge?

> [!NOTE]
> **DOI**: I will use a placeholder DOI `10.1109/ACCESS.2026.XXXXXXX`. You'll need to update this once the paper is assigned an actual DOI.
