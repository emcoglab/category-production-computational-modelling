## Installation and setup

You'll need to Cythonize part of the model code.  See `cognitive_model/README.md` for details.

You'll also need to update paths to point to the referenced data files where appropriate.

## Running

To reproduce the hitrate analysis used in the category production modelling journal paper, run:

```shell script
python 3_category_production_paper_output.py
```

to initially compute the optimal cut-off point.  Then run:

```shell script
python 3_category_production_paper_output.py --manual-cut-off 305
```

to apply that cut-off to the model output.
