# Machine Learning Assignments

Solution code for the two course assignments: regression on the UCI
Superconductivity dataset (Oblig 1) and Decision Trees / Support Vector
Machines on the UCI Adult dataset (Oblig 2).

## Layout
- `Oblig1/` - Oblig 1: Regression on the Superconductivity dataset
  - `scripts/` - one script per question (data exploration, correlation,
    linear and multiple regression, feature importance, polynomial
    regularization)
  - `utils/` - shared data loading and figure helpers
- `Oblig2/` - Oblig 2: Decision Trees and SVMs on the Adult dataset
  - `exercise1/..exercise5/` - one folder per exercise, one script per question
  - `src/` - shared helpers (data loading, split, preprocessing, metrics, CV)
  - `data/` - raw UCI files
  - `README.md` - assignment-specific notes
  - `requirements.md` - dependencies
- `.gitignore` - ignore rules for caches and generated outputs

Reports, assignment sheets and the example template are kept outside the
repo (`../latex`, `../example_latex`, `../assignment`).

## Running
Scripts can be run from the repo root; each adds its own folder to the
module path:

```bash
python Oblig1/scripts/q1_3_linear_regression.py
python Oblig2/exercise3/q3_1_linear_svm.py
```

Both assignments load the raw UCI data through a shared helper, so
separators, missing-value markers and awkward headers are handled in one
place.