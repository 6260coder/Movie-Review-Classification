Purpose of model:  
Sequence classification

Structure of model:  
A softmax that takes the last relevant output from an RNN for classification

Management of files:  
* **data_helpers.py** holds utility functions regarding data preprocessing.
* **Sequence_Classification_Model.py** holds the structure of the neural network as a class.
* **Train.py** implements the training of the network.
* **tar_to_pickle.py** uses functions in data_helpers to preprocess data and pickle the results in neg_samples_in_indices.pkl, pos_samples_in_indices.pkl, token_to_index_vocab.pkl and vocab_in_embeddings.pkl.

Dataset:  
Movie review dataset offered by Standord University's AI department, which can be downloaded at: http://ai.stanford.edu/~amaas/data/sentiment/.
It comes as a compressed tar archive where positive and negative reviews can be found as text files in two according folders.
Pre-trained word vectors from common crawl using Stanford NLP group's GloVe model, which can be downloaded at https://nlp.stanford.edu/projects/glove/.
The dataset contains 840 billion tokens and is a vocabulary of 2.2 million words represented in 300 dimensional vectors.


