Usage Instructions
Step 1: Run model_predict.py
The model will classify the unknown protein sequence into a specific type of scaffold.
Purpose: This step uses a Protein Language Model (PLM) to rapidly filter out irrelevant proteins and identify candidate scaffolds with matching features.

Step 2: Run align.py
Predict the structure of the candidate scaffold protein and align it with the known SBP (Scaffold-Binding Protein) to calculate the TM-Score between them.
Purpose: Structural alignment validates the functional similarity between the candidate scaffold and the reference SBP.
