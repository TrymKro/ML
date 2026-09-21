# Oblig 2 - Decision Trees and SVMs on the Adult dataset

Course assignment IDATG2208 exploring Decision Trees and Support Vector
Machines on the UCI Adult Census Income dataset.

## Layout
- `data/` - raw UCI files (`adult.data`, `adult.test`, `adult.names`, `Index`, `old.adult.names`)
- `src/` - shared helpers (data loading, column definitions)
- `exercise1/` - Exercise 1: Data Preparation scripts (one per question)

## How to run
```bash
python exercise1/q1_1_missing_values.py
```

Every exercise script loads the raw data through `src.data.load_raw()`, so the
`?` missing-value marker and the awkward `adult.test` header are handled in one
place.