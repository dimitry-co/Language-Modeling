# Language Modeling
This project trains a unigram, bigram and a bigram with add-one smoothing language models and evaluates them on a test corpora. 

## How to Run

1. The files ``preprocessing.py``, ``modelling.py``, ``questions.py`` and the two corpora (``train-Spring2023.txt``, ``test.txt``) must be in your current directory.
2. Run ``python3 questions.py`` in your terminal from the directory
3. The program will output three files:
    * ``output.txt`` has the answers to the questions below
    * ``train-pp.txt`` has the preprocessed Brown training corpus.
    * ``test-pp.txt`` has the preprocessed Brown test corpus.
4. ``output.txt`` containd the answers to the questions

## Corpora
Each file is a collection of texts, one sentence per line. train.txt contains about 100,000 sen-
tences from the NewsCrawl corpus. You will use this corpus to train the language models.
The test corpus test.txt is from the same domain and will be used to evaluate the language
models that you trained.

## PRE-PROCESSING
Prior to training, please complete the following pre-processing steps:
1. Pad each sentence in the training and test corpora with start and end symbols (you can
use \<s> and \<\/s>, respectively).
2. Lowercase all words in the training and test corpora. Note that the data already has
been tokenized (i.e. the punctuation has been split off words).
3. Replace all words occurring in the training data once with the token \<unk>. Every word
in the test data not seen in training should be treated as \<unk>.

## TRAINING THE MODELS
train.txt was used to train the following language models:
1. A unigram maximum likelihood model.
2. A bigram maximum likelihood model.
3. A bigram model with Add-One smoothing.

## QUESTIONS
1. How many word types (unique words) are there in the training corpus? Please include
the end-of-sentence padding symbol \<\/s> and the unknown token \<unk>. Do not in-
clude the start of sentence padding symbol \<\s>.
2. How many word tokens are there in the training corpus? Do not include the start of
sentence padding symbol \<s>.
3. What percentage of word tokens and word types in the test corpus did not occur in
training (before you mapped the unknown words to \<unk> in training and test data)?
Please include the padding symbol \<\/s> in your calculations. Do not include the start
of sentence padding symbol \<s>.
4. Now replace singletons in the training data with \<unk> symbol and map words (in the
test corpus) not observed in training to \<unk>. What percentage of bigrams (bigram
types and bigram tokens) in the test corpus did not occur in training (treat \<unk> as a
regular token that has been observed). Please include the padding symbol \<\/s> in your
calculations. Do not include the start of sentence padding symbol \<s>.
5. Compute the log probability of the following sentence under the three models (ignore
capitalization and pad each sentence as described above). Please list all of the param-
eters required to compute the probabilities and show the complete calculation. Which
of the parameters have zero values under each model? Use log base 2 in your calcula-
tions. Map words not observed in the training corpus to the \<unk> token.
    - I look forward to hearing your reply .
6. Compute the perplexity of the sentence above under each of the models.
7. Compute the perplexity of the entire test corpus under each of the models.