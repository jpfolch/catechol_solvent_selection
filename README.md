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