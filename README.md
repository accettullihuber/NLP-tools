# NLP-tools

In this repository we make the code used for the computations in arxiv:  publicly available. In order to run the code you will need the SimVerb3500.txt dataset (included in this repository for convenience) and the matrix datasets named matrices_1160_arg_obj_context_subj.txt and matrices_1160_arg_subj_context_obj.txt of CITE available from GIJS.
The files are organised as follows:
- txt files containing the definitions of the observables which constitute the vectors we are interested in, these include Linear.txt, Quadratic.txt, Cubic1.txt, Quartic1.txt and Additional1.txt
- MFunctions.py contains all the custom implemented functions
- SimVerb3500.txt is the word pair dataset
- Example_notebook contains examples of how to use most of the functions in MFunctions.py
- Means_and_Balance_Accuracy contains the main results of the paper reguarding the means of cosine distances and balanced accuracies in the lexical class distinction task
- Comparing_hyper_hypo contains the functions used to compare the length of the hypernyms and hyponyms in the hyper/hyponym pairs