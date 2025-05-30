# PSS-Finder
PSS-Finder: a protein language model-based framework for mining privileged scaffolds in synthetic binding protein design

# contents
├── 0data_save
│   ├── dataset.csv (trainning dataset)
│   ├── model_weight
│   └── test.csv
├── 0storage
│   ├── 1select_gomc
│   ├── 2selected_gomc
│   ├── 3selected_gomc_type_csv
│   ├── 4selected_gomc_type_pdb
│   ├── 5reference_sbp
│   ├── 6selected_gomc_tm-score
├── 1protBERT
├── 2model_train
│   ├── continue_train.py
│   ├── embedding.py
│   ├── model.py
│   ├── normal_train.py
│   └── test.py
├── 3ESMFold
└── 4prediction
    ├── align.py
    └── model_predict.py
