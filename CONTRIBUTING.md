# Contributing


## How to contribute
To avoid merge conflicts, we will be using [Pull Requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) 
to contribute to this repo. To make any changes:
- Create a new branch 
- Make the changes to the codebase
- When you are ready to merge, create a pull request. Someone will review your 
changes, and merge them into the main branch.

We encourage you to use the pre-commit discussed above to format code. We also 
encourage type-hinting, although neither are required!

## Project structure

For convenience, every model should be implemented as a subclass of `Model`. We 
demonstrate this in `catechol/models/gp`. Each model must define the following functions:

- `_train(train_X, train_y)` - given a dataframe of training data, fit the underlying model
- `_predict(test_X)` - given unseen `test_X`, return a prediction per point


## Formatting and style

We also provide some formatting pre-push hooks. You don't *need* these, but 
they will automatically format your code when you push to GitHub, which is nice
for enforcing code-style consistency across the project. You can install these with

```bash
pre-commit install
```

You can then also run these manually if you'd like, with

```bash
pre-commit run --all-files --hook-stage pre-push
```