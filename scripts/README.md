# The Catechol Benchmark: `scripts`

This repository contains all of the scripts to run the experiments in the paper. Here,
we briefly list the experiments, and the associated scripts. Any script that is not
listed below is included for demonstration purposes only.

For each script, run `python <script.py> --help` to list the available options.


## Shared arguments

Many of the scripts take the following parameters:
 - `-m`/`--model`: the name of the model used to fit the data and make predictions
 - `-f`/`--featurization`: the name of the solvent featurization
 - `-c`/`--config`: an arbitrary dictionary of additional configuration options

The `--model` provided must be one of the models in `catechol.models.ALL_MODELS`. If
you want to add a custom model, ensure it is added to this dictionary. The currently 
available models are:
```python
"GPModel", "LLMModel", "MLPModel", "BaselineModel", "BaselineGPModel", "LODEModel", "EODEModel", "NODEModel", "NPModel"
```

The `--featurization` must be one of the featurizations corresponding to a lookup table
in `/data/featurization_look_ups/`. To provide your own featurizations, create a lookup
table like the ones in that directory, with the filename `<featurization_name>_lookup.csv`.
The currently available featurizations are:
```python
"acs_pca_descriptors", "drfps_catechol", "fragprints", "spange_descriptors"
```

The `--config` corresponds to additional keyword arguments to be passed into the `Model`.
See the individual model classes to see all of the available arguments that can be passed.

For example, the configuration to an LLM can be provided as below:
```bash
<script>.py -m LLMModel -f smiles -c epochs=400 freeze_backbone=True dropout_head=0.5 time_limit=720 pretrained_model_name=rxnfp-pretrained
```
and the config for a GP can be passed as:
```bash
<script>.py -m GPModel -f spange_descriptors -c multitask=False use_input_warp=True use_separated_kernel=False
```

## Experiment scripts

### Regression (sec 3.2, 3.3)

To evaluate the regression performance of the different models on our dataset, use the
scripts `eval_single_solvents.py` for the single solvent evaluation, and 
`eval_solvent_ramps.py` for the full data evalaution (including mixtures of solvents).

To evaluate the different GP extensions considered in section 3.3, look at the 
config options shown for the GP model in the "Shared arguments" section above.

### Transfer learning (sec 3.4)

To evaluate the regression performance when including the observations from the Ethyl
dataset, run `eval_transfer_learning.py`. This script has an additional boolean
argument, `-t`, that includes/excludes the transfer learning dataset for easier 
comparison.

### Active learning (sec 3.5)

To evaluate the active learning problem, run `eval_active_learning.py`. Whilst this 
script has the same shared arguments as other scripts, it currently only supports
`GPModel`s. This script has an additional argument, `--strategy`, that selects which 
active learning strategy is used when acquiring new points. The options for the strategy
are:
```python
"random", "entropy", "mutual_information"
```

### Bayesian optimization (sec 3.5)

To evaluate the bayesian optimization problems, run `eval_bayes_opt.py` and 
`eval_mo_bayes_opt.py` for the scalarized and multi-objective benchmarks respectively.
As with active learning, this currently only supports the GP model. These scripts also
have a `--strategy` argument, which must be one of:
```python
"random", "ei", "ucb"
```
