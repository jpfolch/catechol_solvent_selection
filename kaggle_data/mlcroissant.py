import mlcroissant as mlc
import itertools

# These values are used by mlcroissant
# to perform the necessary auth to fetch the data
# Provide your Kaggle username and API key
os.environ['CROISSANT_BASIC_AUTH_USERNAME'] = 
os.environ['CROISSANT_BASIC_AUTH_PASSWORD'] = 


# Fetch the Croissant JSON-LD
croissant_dataset = mlc.Dataset('https://www.kaggle.com/datasets/aichemy/catechol-benchmark/croissant/download')

# Check what record sets are in the dataset
record_sets = croissant_dataset.metadata.record_sets
print(record_sets)

# Set the path to the file you'd like to load
file_path = ""

# Fetch the records
record_set = croissant_dataset.records(record_set=file_path)
print("First 5 records:",
  list(itertools.islice(record_set, 5))
)
