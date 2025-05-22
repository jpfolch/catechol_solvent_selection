import pandas as pd

results = pd.DataFrame(columns=["Test solvent", "mse", "nlpd"])
for kernel in ['BoGrape_SP', 'BoGrape_ESP']:
    for split in range(24):
        result = pd.read_csv(f"results/single_solvent/GraphGPModel_{kernel}/split_{split}.csv")
        results = pd.concat((results, result))

    print(f"Kernel: {kernel}")
    print('mse:', results['mse'].mean())
    print('nlpd:', results['nlpd'].mean())