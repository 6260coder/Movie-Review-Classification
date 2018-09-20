
# coding: utf-8

import pickle 
import collections
import numpy as np
import math
"""
Both pos_samples and neg_samples are in form of lists of lists of token strings, such as:
[['this', 'is', 'good'], ['this', 'is', 'bad'], ['this', 'is', 'okay']]
"""

# pads a list to the specified length
def pad_to_length(list_to_pad, length):
    length_diff = length - len(list_to_pad)
    if length_diff > 0:
        list_to_pad += [0] * length_diff
    return list_to_pad

# Reads pickled samples into pos_samples and neg_samples as lists of lists
def load_pickled_raw_data():
    with open("./pickled/pos.pkl", "rb") as in_file:
        pos_samples = list()
        while True:
            try:
                sample = pickle.load(in_file)
                pos_samples.append(sample)
            except EOFError:
                break
            
    with open("./pickled/neg.pkl", "rb") as in_file:
        neg_samples = list()
        while True:
            try:
                sample = pickle.load(in_file)
                neg_samples.append(sample)
            except EOFError:
                break
    return pos_samples, neg_samples

# builds a token-to-index vocab on pos and neg samples by tokens' minimal presence
# only tokens that are present at least min_presence times will be included in the vocab
def build_vocab_by_min_presence(pos_samples, neg_samples, min_presence):
    samples = pos_samples + neg_samples
    counter = collections.Counter()
    # build a counter on the samples
    for sample in samples:
        counter.update(sample)
    # trim the counter by min_presence
    trimmed_counter = collections.Counter()
    for token, count in counter.items():
        if count >= min_presence:
            trimmed_counter[token] = count
    # fill the vocab
    token_to_index_vocab = dict()
    token_to_index_vocab["<unk>"] = 1
    for i, (token, count) in enumerate(trimmed_counter.most_common()):
        # leave 0 for nothing and 1 for <unk>
        token_to_index_vocab[token] = i + 2
    return token_to_index_vocab
    
# builds a token-to-index vocab on pos and neg samples by vocabulary size
# the vocab will only include the first vocab_size many most frequent tokens
def build_vocab_by_size(pos_samples, neg_samples, vocab_size):
    samples = pos_samples + neg_samples
    counter = collections.Counter()
    # build a counter on the samples
    for sample in samples:
        counter.update(sample)
    # fill the vocab
    token_to_index_vocab = dict()
    token_to_index_vocab["<unk>"] = 1
    for i, (token, count) in enumerate(counter.most_common(vocab_size)):
        # leave 0 for nothing and 1 for <unk>
        token_to_index_vocab[token] = i + 2
    return token_to_index_vocab

# an intermediate function that is called by integerize_by_vocab_siz and 
# integerize_by_min_presence to integerize samples
# samples are padded to max_sample_len
def integerize_samples_by_vocab(samples, token_to_index_vocab, max_sample_len):
    samples_in_indices = []
    for sample in samples:
        sample_in_indices = []
        for token in sample:
            if token in token_to_index_vocab.keys():
                sample_in_indices.append(token_to_index_vocab[token])
            else:
                sample_in_indices.append(1)
        sample_in_indices = pad_to_length(sample_in_indices, max_sample_len)
        samples_in_indices.append(sample_in_indices)
    return samples_in_indices
    
# name a size for the vocabulary and this function integerizes the pos and neg
# samples for you
# it also returns a token-to-index vocabulary in form of a dictionary
def integerize_by_vocab_size(pos_samples, neg_samples, vocab_size):
    token_to_index_vocab = build_vocab_by_size(pos_samples, neg_samples, vocab_size)
    samples = pos_samples + neg_samples
    max_sample_len = max(len(sample) for sample in samples)
    pos_samples_in_indices = integerize_samples_by_vocab(pos_samples, 
                                                         token_to_index_vocab, 
                                                         max_sample_len)
    neg_samples_in_indices = integerize_samples_by_vocab(neg_samples, 
                                                         token_to_index_vocab, 
                                                         max_sample_len)
    return pos_samples_in_indices, neg_samples_in_indices, token_to_index_vocab

# name a minimal presence for tokens and this function integerizes the pos and neg
# samples for you
# it also returns a token-to-index vocabulary in form of a dictionary
def integerize_by_min_presence(pos_samples, neg_samples, min_presence):
    token_to_index_vocab = build_vocab_by_min_presence(pos_samples, 
                                                       neg_samples, 
                                                       min_presence)
    samples = pos_samples + neg_samples
    max_sample_len = max(len(sample) for sample in samples)
    pos_samples_in_indices = integerize_samples_by_vocab(pos_samples, 
                                                         token_to_index_vocab, 
                                                         max_sample_len)
    neg_samples_in_indices = integerize_samples_by_vocab(neg_samples, 
                                                         token_to_index_vocab, 
                                                         max_sample_len)
    return pos_samples_in_indices, neg_samples_in_indices, token_to_index_vocab


# In[7]:


# swaps a dictionary's keys and values
def reverse_a_dict(d):
    reversed_dict = dict(zip(d.values(), d.keys()))
    return reversed_dict


# In[ ]:


# restore a sample in integers to its original human-readable form
def restore_an_indexed_sample(sample, token_to_index_vocab):
    index_to_token_vocab = reverse_a_dict(token_to_index_vocab)
    restored_sample = []
    for index in sample:
        if index != 0:
            restored_sample.append(index_to_token_vocab[index])
    return restored_sample


# In[3]:


# process one line in the pretrained word embeddings file and 
# add the result to processed_embedding_dict
# key is a string for the token
# value is a 300 long list of floats that is the embedding for the token
def process_raw_embedding(raw_str, processed_embeddings_dict):
    embedding = []
    raw_list = raw_str.split()    
    for i in range(300):
        embedding.append(float(raw_list.pop()))
    token = "".join(raw_list)
    if token not in list(processed_embeddings_dict.keys()):
        processed_embeddings_dict[token.lower()] = embedding  


# In[9]:


# Reads a batch of lines from the pretrained embeddings file, turns them into
# token string to float embedding pairs and store them into a dictionary
def processed_embeddings_dict_generator(batch_size,
                                        file_dir="./pretrained_embeddings/glove.840B.300d.txt"):
    embeddings_pool_file = open(file_dir, "rt", encoding='UTF-8')
    end_of_file_flag = False
    while True:
        processed_embeddings_dict = {}
        if end_of_file_flag == True:
            break
        num_lines = 0
        while num_lines < batch_size:
            line = embeddings_pool_file.readline()
            if not line:
                end_of_file_flag = True
                break
            process_raw_embedding(line, processed_embeddings_dict)
            num_lines += 1
        print("One batch generated.")
        yield processed_embeddings_dict, end_of_file_flag
        
        
        
        


# In[13]:


# intermediate function called by build_vocab_in_embeddings
# try and find unfound tokens in the vocabulary in the current batch
# generated by processed_embeddings_dict_generator before updating them accordingly
def one_pass(vocab_in_embeddings, processed_embeddings_dict):
    unfound_token_count = 0
    for i in range(2, len(vocab_in_embeddings)):
        # If a token is represented with string rather than embeddings,
        # it hasn't been found yet in previous batches
        if type(vocab_in_embeddings[i]) == type(""):
            if vocab_in_embeddings[i] in list(processed_embeddings_dict.keys()):
                vocab_in_embeddings[i] = processed_embeddings_dict[vocab_in_embeddings[i]]
            else:
                unfound_token_count += 1
    if unfound_token_count == 0:
        print("All tokens found.")
        all_tokens_found_flag = True
    else:
        print("Number of unfound tokens after current pass: {}".format(unfound_token_count))
        all_tokens_found_flag = False
    return all_tokens_found_flag
                


# In[15]:


# intermediate function called by build_vocab_in_embeddings
# assigns unfound tokens each an embedding of zeros
def clean_up(vocab_in_embeddings):
    unfound = [0.0] * 300
    for i in range(2, len(vocab_in_embeddings)):
        if type(vocab_in_embeddings[i]) == type(""):
            vocab_in_embeddings[i] = unfound
    print("Cleaned up.")


# In[14]:


# Tries and find tokens from token_to_index_vocab in each batch generated by 
# processed_embeddings_dict_generator. 
# Adds the embeddings to vocab_in_embeddings if found.
def build_vocab_in_embeddings(token_to_index_vocab, batch_size, num_batches):
    # for padded values for example
    void_input = [0.0] * 300
    # embeddig for the "<unk>" in token_to_index_vocab
    unk = [0.0] * 300
    vocab_in_embeddings = []
    vocab_in_embeddings.append(void_input)
    vocab_in_embeddings.append(unk)
    # Unfound tokens are represented with their token strings
    # rather than embeddings
    for token in list(token_to_index_vocab.keys())[1:]:
        vocab_in_embeddings.append(token)
    generator = processed_embeddings_dict_generator(batch_size)
    num_batch_done = 0
    end_of_file_flag = False
    all_tokens_found_flag = False
    current_pass = 0
    while num_batch_done < num_batches and           end_of_file_flag == False and           all_tokens_found_flag == False:
        processed_embeddings_dict, end_of_file_flag = next(generator)
        current_pass += 1
        print("pass {}:".format(current_pass))
        all_tokens_found_flag = one_pass(vocab_in_embeddings, processed_embeddings_dict)
        num_batch_done += 1
    if end_of_file_flag == True:
        print("End of file reached.")
    if all_tokens_found_flag == False:
        clean_up(vocab_in_embeddings)
    return vocab_in_embeddings


# In[ ]:


# partitions data into training set and development set according to dev_sample_percentage
def partition_data_and_labels(data, labels, dev_sample_percentage):
    assert len(data) == len(labels), "batch_iter: length of data doesn't equal length of labels"
    dev_sample_start = int(float(len(data)) * (1 - dev_sample_percentage))
    return data[:dev_sample_start], data[dev_sample_start:], labels[:dev_sample_start], labels[dev_sample_start:]



# shuffles a data labels pair, returns shuffled np.arrays
def shuffle_data_and_labels(data, labels):
    assert len(data) == len(labels), "shuffle: length of data doesn't equal length of labels"
    data = np.array(data)
    labels = np.array(labels)
    shuffle_indices = np.random.permutation(np.arange(len(data)))
    return data[shuffle_indices], labels[shuffle_indices]

# batch generator
def batch_generator(data, labels, batch_size):
    assert len(data) == len(labels), "batch_iter: length of data doesn't equal length of labels"
    num_of_batches = math.ceil(float(len(data)) / float(batch_size))
    for i in range(num_of_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(data))
        yield i, data[start_index : end_index], labels[start_index : end_index]


# In[ ]:


def load_pickled_data():
    with open("./dataset\minimal_presence_5/neg_samples_in_indices.pkl", "rb") as in_file:
        neg_samples_in_indices = pickle.load(in_file)
    with open("./dataset\minimal_presence_5/pos_samples_in_indices.pkl", "rb") as in_file:
        pos_samples_in_indices = pickle.load(in_file)
    with open("./dataset\minimal_presence_5/token_to_index_vocab.pkl", "rb") as in_file:
        token_to_index_vocab = pickle.load(in_file)
    with open("./dataset\minimal_presence_5/vocab_in_embeddings.pkl", "rb") as in_file:
        vocab_in_embeddings = pickle.load(in_file)
    return pos_samples_in_indices,            neg_samples_in_indices,            token_to_index_vocab,            vocab_in_embeddings

