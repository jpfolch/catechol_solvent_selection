import os
os.chdir('./src')

from catechol.data.data_labels import INPUT_LABELS_SINGLE_SOLVENT
from catechol.data.loader import (
    generate_leave_one_out_splits,
    load_single_solvent_data,
    train_test_split,
)

os.chdir('..')
from catechol.models import LLMModel
from catechol.plots.plot_solvent_prediction import plot_solvent_prediction
from catechol import metrics
import matplotlib.pyplot as plt

model_name = "seyonec/ChemBERTa-zinc-base-v1"#"rxnfp-pretrained"#"sagawa/ReactionT5v1-yield"#"Parsa/Buchwald-Hartwig-Yield-prediction" #

model = LLMModel(model_name = model_name,
                 freeze_backbone = False,
                 learning_rate_backbone = 1e-3,
                 learning_rate_head = 1e-3,
                 dropout_backbone= 0.000001,
                 dropout_head = 0.0000001,
                 use_pooler_output = True,
                 custom_head = None,
                 max_length_padding= None,
                 epochs = 100,
                 use_validation = "leave_one_solvent_out",
                 batch_size = 32
                 )

X, Y = load_single_solvent_data()
# remove unnecessary columns
X = X[['Residence Time', 'Temperature','Reaction SMILES','SOLVENT NAME']]
model.train(train_X = X, train_Y = Y)



split_generator = generate_leave_one_out_splits(X, Y)
# this will generate a new split each time you call `next` on the generator
# you can, instead, use a for loop to iterate over split_generator
(train_X, train_Y), (test_X, test_Y) = next(split_generator)

model.train(train_X = train_X, train_Y = train_Y)



train_X, test_X = train_test_split(X, train_percentage=0.8, seed=1)
train_Y, test_Y = train_test_split(Y, train_percentage=0.8, seed=1)
predictions = model.predict(test_X)
print(predictions)

# calculate some metrics
mse = metrics.mse(predictions, test_Y)
nlpd = metrics.nlpd(predictions, test_Y)
print(f"{mse=}, {nlpd=}")

# plot the predictions
plot_solvent_prediction(model, test_X, test_Y)
(train_X, train_Y), (test_X, test_Y) = next(split_generator)
plot_solvent_prediction(model, test_X, test_Y)
(train_X, train_Y), (test_X, test_Y) = next(split_generator)
plot_solvent_prediction(model, test_X, test_Y)

plt.show()