
# coding: utf-8


"""
Reads the tar archive, picks out positive and negative examples and preprocesses (tokenization and lowercasing) 
them before dumping them into pos.pkl and neg.pkl respectively.
Each resulting pickle file holds lists of strings, one list per example.
"""

import tarfile
import pickle

import re
TOKEN_REGEX = re.compile(r'[A-Za-z]+|[!?.:,()]')

pos_pickle_file = open("./pickled/pos.pkl", "wb")
neg_pickle_file = open("./pickled/neg.pkl", "wb")

def send_to_pickle(file_name, pickle_file, i):
    file_reader = archive.extractfile(file_name)
    read_bytes = file_reader.read()
    text_str = read_bytes.decode()
    data_sample = TOKEN_REGEX.findall(text_str)
    data_sample = [token.lower() for token in data_sample]
    pickle.dump(data_sample, pickle_file)

archive_dir = "./dataset/aclImdb_v1.tar.gz"
archive = tarfile.open(archive_dir)
for i, file_name in enumerate(archive.getnames()):
    if file_name.startswith("aclImdb/train/pos/"):
        send_to_pickle(file_name, pos_pickle_file, i)
        print("file number {} sent to pos_pickle".format(i))
    elif file_name.startswith("aclImdb/train/neg/"):
        send_to_pickle(file_name, neg_pickle_file, i)
        print("file number {} sent to neg_pickle".format(i))
archive.close()      
pos_pickle_file.close()
neg_pickle_file.close()
print("Done pickling.")