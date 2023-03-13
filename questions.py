from modelling import *
import pandas as pd
from IPython.display import display

# Preprocessing and Training/displaying Models
[train_word_freq_before_unk, train_word_freq_with_unk, train_words_with_unk] = \
    preprocess('train-Spring2023.txt', 'train-pp.txt', None)
[test_word_freq_before_unk, test_word_freq_with_unk, test_words_with_unk] = \
    preprocess('test.txt', 'test-pp.txt', train_word_freq_with_unk)

#Modelling ngrams
unigram = unigram_model(train_word_freq_with_unk)
bigram_model = train_bigram(train_words_with_unk, train_word_freq_with_unk, False)
bigram_add1_model = train_bigram(train_words_with_unk, train_word_freq_with_unk, True)

output = open('output.txt', 'w')

# Questions
output.write('Question 1\n'
            'Number of word types in training corpus excluding the <s> symbol: ' +
            str(len(train_word_freq_with_unk) - 1) + '\n')

output.write('\nQuestion 2.\n'
            'Total number of word tokens in the training corpus excluding <s> symbol: ' +
            str(sum(train_word_freq_with_unk.values()) - 100000) + '\n')

unseen_words = get_percent_of_unseen_words(train_word_freq_before_unk, test_word_freq_before_unk)
output.write('\nQuestion 3.\n'
            'Percentage of word tokens in test corpus that did not occur in the training: ' +
            str(unseen_words[0]) + '%\n')
output.write('Percentage of word types in test corpus that did not occur in the training: '
      + str(unseen_words[1]) + '%\n')

unseen_bigrams = get_percent_of_unseen_bigrams(train_words_with_unk, test_words_with_unk)
output.write('\nQuestion 4.\n'
            'Percentage of bigram tokens in test corpus that did not occur in the training: ' +
            str(unseen_bigrams[0]) + '%\n')
output.write('Percentage of bigram types in test corpus that did not occur in the training: ' +
            str(unseen_bigrams[1]) + '%\n')

output.write('\nQuestions 5 and 6:\n')
sentence = 'I look forward to hearing your reply . '
output.write('the sentence: ' + sentence + '\n')

unigram_computation = unigram_prob_computations(sentence, unigram, train_word_freq_with_unk)
output.write('\n' + unigram_computation + '\n')

bigram_computations = bigram_mle_computations(sentence, bigram_model, train_word_freq_with_unk)
output.write(bigram_computations + '\n')

bigram_add1_computes = bigram_add1_computations(sentence, bigram_add1_model, train_word_freq_with_unk)
output.write(bigram_add1_computes + '\n')

output.write('\nQuestion 7:\n')

unigram_perp_of_corpus = compute_unigram_perplexity(test_words_with_unk, unigram)
corpus_bigram_perpl = compute_bigram_perplexity(test_words_with_unk, bigram_model)
corpus_bigram_add1_perpl = compute_bigram_perplexity(test_words_with_unk, bigram_add1_model)

output.write('The perplexity of the test corpus under the Unigram model: ' + 
            str(unigram_perp_of_corpus) + '\n')
output.write('Perplexity of test corpus under the Bigram MLE model: ' +
            str(corpus_bigram_perpl) + '\n')
output.write('Perplexity of test corpus under the Bigram Add-One model: ' +
            str(corpus_bigram_add1_perpl))