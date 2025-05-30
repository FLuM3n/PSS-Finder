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
