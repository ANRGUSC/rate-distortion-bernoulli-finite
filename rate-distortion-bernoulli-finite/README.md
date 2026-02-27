# Finite Block Length Rate-Distortion Theory for the Bernoulli Source

A self-contained LaTeX tutorial on rate-distortion theory for a Bernoulli(p) source with Hamming distortion, covering the classical rate-distortion function, the Blahut-Arimoto algorithm, and finite block length refinements including the rate-distortion dispersion and normal approximation.

## Repository Structure

```
├── main.tex                    # Tutorial document (LaTeX)
├── references.bib              # Bibliography
├── figures/                    # Generated figure PDFs
├── scripts/
│   ├── rate_distortion.py      # Binary entropy and R(D) curves
│   ├── blahut_arimoto.py       # Blahut-Arimoto algorithm
│   ├── dispersion.py           # d-tilted information and V(D)
│   ├── finite_blocklength.py   # Finite-n bounds and normal approximation
│   └── generate_all_figures.py # Master script to regenerate all figures
└── requirements.txt            # Python dependencies
```

## Generating Figures

```bash
pip install -r requirements.txt
cd scripts
python generate_all_figures.py
```

All figures are saved as PDF in the `figures/` directory.

## Building the PDF

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Contents

1. **Introduction** — motivation and overview
2. **Probability and Information Foundations** — entropy, mutual information, typical sequences
3. **The Rate-Distortion Problem** — lossy compression, distortion measures, test channels
4. **Rate-Distortion for the Bernoulli Source** — closed-form R(D) = H(p) - H(D)
5. **The Blahut-Arimoto Algorithm** — iterative computation with convergence analysis
6. **Finite Block Length Theory** — d-tilted information, dispersion V(D), normal approximation
7. **Numerical Explorations** — comprehensive figures and validation
8. **Conclusion** — summary and open problems
