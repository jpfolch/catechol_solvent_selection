import numpy as np
import pandas as pd

from catechol import metrics
from catechol.data.data_labels import INPUT_LABELS_GRAPH_GP, TARGET_LABELS
from catechol.data.loader import (
    generate_leave_one_graph_out_splits,
    load_single_solvent_data,
)
from catechol.models import GraphGPModel

results = []
for kernel in ["BoGrape_SP", "BoGrape_ESP", "RW", "SW", "WL"]:
    model = GraphGPModel(target_labels=TARGET_LABELS, kernel_name=kernel)    # Implemented kernels: "BoGrape_SP", "BoGrape_ESP", "RW", "SW", "WL"
    X, Y = load_single_solvent_data()
    # remove unnecessary columns
    X = X[INPUT_LABELS_GRAPH_GP]
    Y = Y[TARGET_LABELS]

    # you can also use leave-one-out splits of the data
    split_generator = generate_leave_one_graph_out_splits(X, Y)
    # this will generate a new split each time you call `next` on the generator
    # you can, instead, use a for loop to iterate over split_generator
    next(split_generator)
    (train_X, train_Y), (test_X, test_Y) = next(split_generator)
    model.train(train_X, train_Y)

    predictions = model.predict(test_X)

    # calculate some metrics
    mse = metrics.mse(predictions, test_Y)
    nlpd = metrics.nlpd(predictions, test_Y)
    # print(f"{mse=}, {nlpd=}")
    results.append([mse, nlpd])

results_df = pd.DataFrame(np.array(results).T, index=["mse", "nlpd"], columns=["BoGrape_SP", "BoGrape_ESP", "RW", "SW", "WL"])
print(results_df)

