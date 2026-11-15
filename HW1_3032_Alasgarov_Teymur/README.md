# DLE-AI-202 Deep Learning: Homework 1 (Weeks 1-2), PyTorch

**Neural-network foundations: tensors, losses, backpropagation, initialisation, training loops, optimisers, regularisation.**

Read `statement.pdf` first. This folder is the starter kit.

## What you do
1. **Part A, written (30)**: eight problems, answered in `report_template.tex`.
2. **Part B, code (45)**: complete the stubs in `src/` until `pytest` is green.
3. **Part E, experiments (15)**: four experiments in `hw1_notebook.ipynb`.
4. **Part C, reflection (10)** and **bonus (+10)**.

The data (scikit-learn digits) is built in. No GPU needed.

## Layout
```
statement.pdf / statement.tex   the assignment
hw1_notebook.ipynb              your working notebook (tests + experiments)
torch_cheatsheet.ipynb          runnable PyTorch reference: keep it open
src/                            stubs (B1-B5, bonus) and provided helpers (data.py, plotting.py)
tests/                          pytest checks: your own feedback
report_template.tex             fill in, compile to report.pdf
figures/                        final_curves.png is written here
pledge.txt                      sign it; declare AI use
requirements.txt
```

## Run
```bash
pip install -r requirements.txt
pytest                              # tests fail until you implement the stubs
STUDENT_ID=12345 jupyter lab        # your Academy ID seeds the split and initialisation
tectonic report_template.tex        # or pdflatex, twice
```
Tiny networks are fastest on one CPU thread, so `src/data.py` sets one. Override with `DL_HW_THREADS=4`.

## Submit
`HW1_<STUDENT_ID>.zip` on Moodle, containing `report.pdf`, `src/`, `hw1_notebook.ipynb` (with outputs: Restart & Run All first), `figures/final_curves.png`, and a signed `pledge.txt`.
Released Sunday 1 November 2026. Due **Sunday 15 November 2026, 23:59 Baku time** (two weeks).
