# TODO: replace this with actual data

import pandas as pd

from catechol.models import GPModel


model = GPModel()
train_X = pd.DataFrame({"x1": [0.1, 0.2, 0.3], "x2": [0.4, 0.5, 0.6]})
train_Y = pd.DataFrame({"Product 1 yield": [0.1, 0.2, 0.3], "Product 2 yield": [0.4, 0.5, 0.6]})
model.train(train_X, train_Y)

test_X = pd.DataFrame({"x1": [0.15, 0.25], "x2": [0.51, 0.7]})
predictions = model.predict(test_X)
print(predictions)