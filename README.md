# catechol_solvent_selection
Repository for the code and data on the catechol dataset for solvent selection and machine learning.


### Installation

The requirements for this project can be installed using:
```bash
pip install -e .
```
or
```bash
pip install -r requirements.txt --no-deps
```

We recommend installing these in a fresh virtual environment, with Python version
3.11 or greater. We use [`uv`](https://docs.astral.sh/uv/) to manage dependencies, 
but you don't need to install this yourself. If you wish to add more packages to
the requirements, and you don't have `uv` installed, create a PR/issue.


## Dataset

Here, we provide a brief overview of the dataset in this repo.

The dataset is divided into two:
- `catechol_single_solvent_yields` contains only the single-solvent data
- `catechol_full_data_yields` contains the full data set with mixture solvents

We also provide some pre-computed featurizations, which can be looked up with the 
`SOLVENT NAME` column.

### Single solvent columns
Below is a table of all the columns in the `catechol_single_solvent_yields` csv:

| Name | Type | Description |
|--------|--------|--------|
| `EXP NUM` | int| Experiment index; all rows with the same `EXP NUM` will use the same solvent|
| `Residence Time` | float | Time (in minutes) of the reaction|
| `Temperature`| float | Temperature (in Celsius) of the reaction|
| `SM` | float | Quantity of starting material measured (yield %)|
| `Product 2` | float | Quantity of product 2 measured (yield %)| 
| `Product 3` | float | Quantity of product 3 measured (yield %)| 
| `SOLVENT_NAME` | str | Chemical name of the solvent; used as a key when looking up featurizations| 
| `SOLVENT_RATIO` | list[float] | Ratio of component solvents [1]|
| `{...} SMILES` | str | SMILES string representation of a molecule|

[1] This is different than the ratios in the solvent ramp experiments. Here, a single solvent has two component molecules, eg. the solvent "Acetonitrile.Acetic Acid" has two compounds. The `SOLVENT_RATIO` gives the ratio between these compounds. Most solvents consist of only a single compound, so the ratio will be `[1.0]`.

**Inputs**: `Residence Time`, `Temperature`, `SOLVENT NAME`

**Outputs**: `SM`, `Product 2`, `Product 3` 

### Full data columns

The full data contains some additional columns, since these experiments ramp between
two solvents:

| Name | Type | Description |
|--------|--------|--------|
| `SolventB%` | float | Percent concentration of solvent B; the rest of the solvent is made up of solvent A|
| `SOLVENT {A/B} NAME` | str | Chemical name of the solvents; used as a key when looking up featurizations|

**Inputs**: `Residence Time`, `Temperature`, `SOLVENT A NAME`, `SOLVENT B NAME`, `SolventB%`

**Outputs**: `SM`, `Product 2`, `Product 3` 


## Running experiments

All of the scripts used to carry out the experiments are in `scripts/`, including
- `eval_solvent_ramps.py`
- `eval_single_solvents.py`
- `eval_transfer_learning.py`
- `eval_active_learning.py`
- `eval_bayes_opt.py`

Each of these files takes, as argument, a model and a featurization, as well
as an additional configuration string. See the [script README](./scripts/README.md), as
well as the individual scripts, for further details.

## Referencing

If you use this datatset, please cite us as below:

```bibtex
@InProceedings{boyne2025catechol,
  title     = {The Catechol Benchmark: Time-series Solvent Selection Data for Few-shot Machine Learning},
  booktitle = {Advances in Neural Information Processing Systems},
  author    = {Boyne, Toby and Campos, Juan S. and Langdon, Becky D. and Qing, Jixiang and Xie, Yilin and Zhang, Shiqiang and Tsay, Calvin and Misener, Ruth and Davies, Daniel W. and Jelfs, Kim E. and Boyall, Sarah and Dixon, Thomas M. and Schrecker, Linden and Folch, Jose Pablo},
  year      = {2025},
}
```