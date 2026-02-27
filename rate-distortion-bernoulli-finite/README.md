# Finite Block Length Rate-Distortion Theory for the Bernoulli Source

A self-contained tutorial on rate-distortion theory for a Bernoulli(*p*) source with Hamming distortion. The tutorial develops the classical rate-distortion function *R*(*D*) = *H*(*p*) - *H*(*D*), the Blahut-Arimoto algorithm, and finite block length refinements including the *d*-tilted information, rate-distortion dispersion, and the normal approximation from Kostina and Verdu (2012).

All mathematical derivations are accompanied by Python scripts that generate the figures and independently verify the formulas.

## Repository Structure

```
main.tex                        LaTeX source for the tutorial
main.pdf                        Compiled PDF (24 pages)
references.bib                  Bibliography (12 references)
LICENSE.md                      PolyForm Noncommercial 1.0.0
figures/                        Pre-generated figure PDFs (10 figures)
scripts/
  rate_distortion.py            Binary entropy, R(D) curves (Figs 1-2)
  blahut_arimoto.py             Blahut-Arimoto algorithm (Figs 3-4)
  dispersion.py                 d-tilted information, V(D) (Figs 5-6)
  clt_histogram.py              CLT histogram of d-tilted information (Fig 7)
  finite_blocklength.py         Normal approximation, finite-n bounds (Figs 8-10)
  generate_all_figures.py       Master script to regenerate all figures
requirements.txt                Python dependencies (numpy, scipy, matplotlib)
```

## Tutorial Contents

1. **Introduction** -- Motivation via a concrete *n*=2 example; the gap between finite block length and the Shannon limit.
2. **Probability and Information Foundations** -- Entropy, mutual information, typical sequences.
3. **The Rate-Distortion Problem** -- Lossy compression setup, Hamming distortion, test channels, Shannon's single-letter formula.
4. **The Rate-Distortion Function for the Bernoulli Source** -- Two derivations of the optimal test channel (Lagrangian/KKT and entropy maximization); closed-form *R*(*D*) = *H*(*p*) - *H*(*D*).
5. **The Blahut-Arimoto Algorithm** -- Iterative computation with convergence demonstration.
6. **Beyond the Asymptotic Limit: Finite Block Length** -- *d*-tilted information, dispersion *V*(*D*), the normal approximation *R*(*n*, *D*, *eps*) = *R*(*D*) + sqrt(*V*(*D*)/*n*) *Q*^{-1}(*eps*) + *O*(log *n* / *n*).
7. **Numerical Explorations** -- Ten figures with cross-references to the generating scripts.
8. **Conclusion** -- Summary of key results.

## Generating Figures

```bash
pip install -r requirements.txt
python scripts/generate_all_figures.py
```

All ten figures are saved as PDF in the `figures/` directory. The scripts also run internal verification checks (e.g., confirming E[j_X(X, D)] = R(D) to machine precision).

## Building the PDF

Requires a LaTeX distribution with `pdflatex` and `bibtex`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## License

This work is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). You may use, modify, and distribute this work for any noncommercial purpose. See `LICENSE.md` for the full terms.
