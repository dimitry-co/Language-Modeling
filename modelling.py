from preProcessing import *
from collections import Counter
import math

def unigram_model(train_dict):
    unigram = train_dict.copy()
    unigram.pop('<s>')
    n = sum(unigram.values())
    unigram = {word: count/n for word, count in unigram.items()}
    return unigram

from collections import Counter
from typing import Dict, List

#returns a list of of bigrams
def generate_bigrams(words):
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append((words[i], words[i+1]))
    return bigrams

def make_count_matrix(train_text, word_freq_dict, add_one):
    vocabulary = list(set(word_freq_dict))

    bigrams = generate_bigrams(train_text)
    bigram_counts = Counter(bigrams)

    col_index = {word:i for i, word in enumerate(vocabulary)}
    row_index = col_index.copy()

    nrow = len(row_index)
    ncol = len(col_index)
    count_matrix = np.zeros((nrow, ncol))

    bigram_starts_with_end_symbol = 0
    for bigram, count in bigram_counts.items():
        prev_word, curr_word = bigram
        i = row_index[prev_word]
        j = col_index[curr_word]
        count_matrix[i, j] = count
    if add_one:
        count_matrix += 1

    count_matrix = pd.DataFrame(count_matrix, index=row_index, columns=vocabulary)
    return count_matrix

def train_bigram(train_text, word_freq_dict, add_one_smoothing):
    count_matrix = make_count_matrix(train_text, word_freq_dict, add_one_smoothing)
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return prob_matrix

# get percent of word tokens and word types in the test corpus thats not in the training corpus
def get_percent_of_unseen_words(train_word_dict, test_word_dict):
    total_unseen_tokens = 0
    total_unseen_types = 0
    number_of_tokens_in_test = sum(test_word_dict.values()) - test_word_dict['<s>']
    number_of_types_in_test = len(test_word_dict) - 1
  
    for word in test_word_dict:
        if word not in train_word_dict:
            total_unseen_tokens += test_word_dict[word]
            total_unseen_types += 1
    percent_unseen_tokens = round((total_unseen_tokens / number_of_tokens_in_test * 100), 2)
    percent_unseen_types =  round((total_unseen_types / number_of_types_in_test * 100), 2)
    return [percent_unseen_tokens, percent_unseen_types]

def get_percent_of_unseen_bigrams(train_text, test_text):
    total_unseen_bigram_tokens = 0
    total_unseen_bigram_types = 0
    train_bigrams = generate_bigrams(train_text)
    train_bigram_dict = Counter(train_bigrams)
    
    test_bigrams = generate_bigrams(test_text)
    test_bigram_dict = Counter(test_bigrams)
    
    bigram_tokens_in_test = sum(test_bigram_dict.values())
    bigram_types_in_test = len(test_bigram_dict)
    
    for bigram in test_bigram_dict:
        if bigram not in train_bigram_dict:
            total_unseen_bigram_tokens += test_bigram_dict[bigram]
            total_unseen_bigram_types += 1
    percent_unseen_bigram_tokens = round((total_unseen_bigram_tokens / bigram_tokens_in_test * 100), 2)
    percent_unseen_bigram_types = round((total_unseen_bigram_types / bigram_types_in_test * 100), 2)
    return [percent_unseen_bigram_tokens, percent_unseen_bigram_types]


# ompute log probabilities and perplexity for the three models
def unigram_prob_computations(sentence, unigram, train_freq):
    unigram_compute = 'model: Unigram\n'
    text = assign_unk_to_words(train_freq, (sentence.lower() + '</s>').split())
    unigram_compute += 'The computing sentence:\t' + str(text) + '\n'

    word_probs = {word: unigram[word] for word in text}
    df = pd.DataFrame.from_dict(word_probs, orient='index', columns=['probabilities'])
    unigram_compute += '\nParameters required to compute the log probability:\n' + str(df) +'\n'
    
    log_prob_of_each_word = {word : math.log(prob, 2) for word, prob in word_probs.items()}
    df = pd.DataFrame.from_dict(log_prob_of_each_word, orient='index', columns=['log prob'])
    unigram_compute += '\nThe log base 2 of the probabilities of each word:\n' + str(df) + '\n'

    log_prob = sum(log_prob_of_each_word.values())
    unigram_compute += '\nThe log probability of the sentence: ' + str(log_prob)
    
    N = len(text)
    perplexity = 2 **(-log_prob / N)  
    unigram_compute += '\nThe perplexity of the sentence: ' + str(perplexity) + '\n'
    
    return unigram_compute
        
def bigram_mle_computations(sentence, bigram_model, train_freq):
    compute = 'model: Bigram mle\n'
    tokens = tokenize_sentences([sentence])
    text = assign_unk_to_words(train_freq, tokens)
    compute += 'The Computing sentence:\t' + str(text) + '\n'
    
    bigrams_in_text = generate_bigrams(text)
    bigram_probs = {bigram: bigram_model.loc[bigram] for bigram in bigrams_in_text}
    df = pd.DataFrame.from_dict(bigram_probs, orient='index', columns=['probabilities'])
    compute += '\nbigrams required to compute the log probability:\n' + str(df) + '\n'
    
    has_zeros = False
    log_prob_of_bigrams = {}
    for bigram, prob in bigram_probs.items():
        if prob <= 0:
            log_prob_of_bigrams[bigram] = float('-inf')
            has_zeros = True
        else:
            log_prob_of_bigrams[bigram] = math.log(prob, 2)

    if has_zeros:
        compute += '\nThe log probability is undefined for the following bigrams that are unseen in the bigram model:\n'
        for bigram, prob in bigram_probs.items():
            if prob == 0:
                compute += '"' + str(bigram[0]) + ' ' + str(bigram[1]) + '"\n'
    else:
        df = pd.DataFrame.from_dict(log_probs_of_bigrams, orient='index', columns=['log prob.'])
        compute += '\nThe log base 2 of each bigram probability:\n' + str(df) + '\n'
        log_prob = sum(log_prob_of_bigrams.values())
        compute += '\nThe bigram log probability of the sentence:' + str(log_prob)
        
        N = len(text)
        perplexity = 2 ** (-log_prob / N)
        compute += '\nThe perplexity of the sentence: ' + str(perplexity)
    return compute   

def bigram_add1_computations(sentence, bigram_model, train_freq):
    compute = 'model: Bigram Add-One\n'
    tokens = tokenize_sentences([sentence])
    text = assign_unk_to_words(train_freq, tokens)
    compute += 'The Computing sentence:\t' + str(text) + '\n'
    
    bigrams_in_text = generate_bigrams(text)
    bigram_probs = {bigram: bigram_model.loc[bigram] for bigram in bigrams_in_text}
    df = pd.DataFrame.from_dict(bigram_probs, orient='index', columns=['probabilities'])
    compute += '\nbigrams required to compute the log probability:\n' + str(df) + '\n'
    
    log_prob_of_bigrams = {bigram : math.log(prob, 2) for bigram, prob in bigram_probs.items()}
    df = pd.DataFrame.from_dict(log_prob_of_bigrams, orient='index', columns=['log prob.'])
    compute += '\nThe log base 2 of each bigram probability:\n' + str(df) + '\n'
    
    log_prob = sum(log_prob_of_bigrams.values())
    compute += '\nThe bigram log probability of the sentence: ' + str(log_prob)
        
    N = len(text)
    perplexity = 2 ** (-log_prob / N)
    compute += '\nThe perplexity of the sentence: ' + str(perplexity)
    return compute   

def compute_unigram_perplexity(corpus, unigram):
    corpus_vocab = set(corpus)
    log_prob_of_words_in_corpus = {word: math.log(unigram[word], 2) for word in corpus_vocab if word != '<s>'}
    log_prob_of_corpus = 0
    N = 0
    for word in corpus:
        if word != '<s>':
            log_prob_of_corpus += log_prob_of_words_in_corpus[word]
            N += 1        
    perplexity = 2 ** (-log_prob_of_corpus / N)
    return perplexity

def compute_bigram_perplexity(corpus, bigram_model):
    bigrams_in_corpus = generate_bigrams(corpus)
    has_zeros = False
    log_prob_of_bigrams = {}
    bigram_log_prob_2 = 0
    for bigram in bigrams_in_corpus:
        if bigram_model.loc[bigram] <= float(0):
            log_prob_of_bigrams[bigram] = float('-inf')
            has_zeros = True
        else:
            log_prob_of_bigrams[bigram] = math.log(bigram_model.loc[bigram], 2)

    if has_zeros:
        perplexity = 'undefined'
    else:
        bigram_log_prob = sum(log_prob_of_bigrams.values())
        N = len(corpus)
        perplexity = 2 ** (-bigram_log_prob / N)
        
    return perplexity