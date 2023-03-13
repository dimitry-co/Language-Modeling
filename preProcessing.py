import numpy as np
import pandas as pd
from collections import Counter

def tokenize_sentences(sentences):  #returns a concatenated list of words lowercased with padded symbols
    pr_txt = ""
    for sentence in sentences:
        pr_txt +=(' <s> '+sentence.lower()+' </s> ')
    return pr_txt.split()

def lower_and_padding_two(sentences):  #returns a list of words lowercased with padded symbols
    processed_text = ""
    for sentence in sentences:
        processed_text += " <s> " + sentence.lower() + " </s> "
    return processed_text.split()

def count_words(words):
    word_counts = {}
    for word in words:
        if word not in word_counts.keys(): # if word not in the dict yet, set count to 1
            word_counts[word] = 1
        else:
            word_counts[word] += 1
    return word_counts

def assign_unk_to_train_words(word_counts, train_words): # replace every word in the training corpus replace with unk
    train_words_unk = train_words
    for i in range(len(train_words)): 
        if word_counts[train_words[i]] == 1:
            train_words_unk[i] = '<unk>'       

def assign_unk_to_words(trained_word_counts, words):
    for i in range(len(words)): 
        if words[i] not in trained_word_counts: # word is not a key in the trained word count dict
            words[i] = '<unk>'
    return words

def preprocess(file_read, file_write, train_word_freq):
    with open(file_read, 'r') as f:
        data = f.readlines()
    
    words = tokenize_sentences(data)
    freq_before_unk = count_words(words)

    if train_word_freq is None: #training corpus
        assign_unk_to_train_words(freq_before_unk, words)
    else:
        words = assign_unk_to_words(train_word_freq, words)

    freq_with_unk = count_words(words)

    write_file = open(file_write, 'w')
    text = ' '.join(words)
    write_file.write(text)
    write_file.close()

    return [freq_before_unk, freq_with_unk, words]