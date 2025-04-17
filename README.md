# catechol_solvent_selection
Repository for the code and data on the catechol dataset for solvent selection and machine learning.


### Installation

The requirements for this project can be installed using:

```bash
pip install -r requirements.txt
```

We recommend installing these in a fresh virtual environment, with Python version
3.11 or greater. We use [`uv`](https://docs.astral.sh/uv/) to manage dependencies, 
but you don't need to install this yourself. If you wish to add more packages to
the requirements, and you don't have `uv` installed, create a PR/issue.

We also provide some formatting pre-push hooks. You don't *need* these, but 
they will automatically format your code when you push to GitHub, which is nice
for enforcing code-style consistency across the project. You can install these with

```bash
pre-commit install
```

### Contributing
To avoid merge conflicts, we will be using [Pull Requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) 
to contribute to this repo. To make any changes:
- Create a new branch 
- Make the changes to the codebase
- When you are ready to merge, create a pull request. Someone will review your 
changes, and merge them into the main branch.

We encourage you to use the pre-commit discussed above to format code. We also 
encourage type-hinting, although neither are required!

### Project structure

For convenience, every model should be implemented as a subclass of `Model`. We 
demonstrate this in `catechol/models/gp`. Each model must define the following functions:

- `_train(train_X, train_y)` - given a dataframe of training data, fit the underlying model
- `_predict(test_X)` - given unseen `test_X`, return a prediction per point
- `_ask()` - propose a new experiment (for Bayesian optimization)

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
