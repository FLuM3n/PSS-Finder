***PSS-Finder***
=
a protein language model-based framework for mining privileged scaffolds in synthetic binding protein design
-
![Workflow of PSS-Finder](PSS-Finder.png)

# Introduction  
PSS-Finder is a framework designed to predict whether novel protein sequences belong to 53 categories of synthetic binding protein scaffolds. The framework provides pre-trained weights, enabling users to directly load them for prediction. It also supports model training from scratch by modifying the training dataset.

# Function  
The framework utilizes the pre-trained protein language model (PLM) protBERT to process protein sequences, generating two types of embeddings as input features for the neural network. The classification model computes probability scores for each scaffold category and assigns the sequence to the class with the highest probability.  
After classification, the framework automatically invokes ESMFold for rapid structure prediction. The predicted structures are aligned against known synthetic protein scaffolds, and sequences with a TM-Score ≥ 0.5 are recorded as potential synthetic binding protein scaffolds, followed by comprehensive annotation output.

# contents
const PSSFinder = (function() {
  // Core structure
  const structure = {
    data_save: {
      dataset: "dataset.csv", // training dataset
      model_weight: {},
      test: "test.csv"
    },
    storage: {
      select_gomc: {},       // 1select_gomc
      selected_gomc: {},     // 2selected_gomc
      gomc_type_csv: {},    // 3selected_gomc_type_csv
      gomc_type_pdb: {},    // 4selected_gomc_type_pdb
      reference_sbp: {},     // 5reference_sbp
      gomc_tm_score: {}     // 6selected_gomc_tm-score
    },
    protBERT: {},            // 1protBERT
    model_train: {
      continue_train: function() {},  // continue_train.py
      embedding: function() {},       // embedding.py
      model: function() {},           // model.py
      normal_train: function() {},    // normal_train.py
      test: function() {}             // test.py
    },
    ESMFold: {},             // 3ESMFold
    prediction: {
      align: function() {},   // align.py
      model_predict: function() {} // model_predict.py
    }
  };

  // Public API
  return {
    init: function() {
      console.log("PSS-Finder initialized");
      return this;
    },
    
    getStructure: function() {
      return structure;
    },
    
    train: function(mode = "normal") {
      if (mode === "normal") {
        this.model_train.normal_train();
      } else {
        this.model_train.continue_train();
      }
      return this;
    },
    
    predict: function() {
      this.prediction.model_predict();
      return this;
    }
  };
})();

// Usage example
const pss = PSSFinder.init();
console.log(pss.getStructure());
pss.train().predict();
