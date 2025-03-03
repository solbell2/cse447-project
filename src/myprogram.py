#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pandas as pd
import pickle

def process_text_for_Ngram(sents, N: int = 2):

    """
    Adds N-1 <sos> tokens to the start of every sentence in the text.

    Inputs:
        - sents: List[str], List of sentences
        - N: int, the N in N-gram

    Outputs:
        - List[str], the processed text
    """
    processed_sents = None

    # YOUR CODE HERE
    processed_sents = []
    for sentence in sents:
      new_sent = []
      for i in range(N-1):
        new_sent.append('<sos> ')
      for letter in sentence:
        new_sent.append(letter)
      processed_sents.append(new_sent)

    return processed_sents

class WordNGramLM:

    def __init__(self, N: int):
        self.N = N
        self.counts = {}
        self.vocab = set()
        self.total_counts = {}
        self.all_words = {}
        self.all_probs = {}

    def fit(self, train_data):

        """
        Trains an N-gram language model.

        Inputs:
            - train_data: str, sentences in the training data

        """

        # YOUR CODE HERE
        new_train_data = process_text_for_Ngram(train_data, self.N)

        for sentence in new_train_data:
          wordsLen = len(sentence)
          for letter in sentence:
            self.vocab.add(letter)
          for i in range(wordsLen - self.N + 1):
            key = "".join(sentence[i:i + self.N - 1])
            value = sentence[i + self.N - 1]
            if key in self.counts:
              if value in self.counts[key]:
                self.counts[key][value] = self.counts[key][value] + 1
              else:
                self.counts[key][value] = 1
            else:
              self.counts[key] = {}
              self.counts[key][value] = 1

        for key in self.counts.keys():
          self.total_counts[key] = np.sum(list(self.counts[key].values()))
          self.all_words[key] = [*self.counts[key]]
          self.all_probs[key] = np.divide(np.asarray(list(self.counts[key].values())), self.total_counts[key])

    def sample_text(self, prefix: str = "<sos>", max_words: int = 100):

        """
        Samples text from the N-gram language model.
        Terminate sampling when either max_words is reached or when <eos> token is sampled.
        Inputs:
            - prefix: str, the prefix to start the sampling from. Can also be multiple words separated by spaces.
            - max_words: int, the maximum number of words to sample

        Outputs:
            - str, the sampled text

        Note: Please use np.random.choice for sampling next words
        """

        # YOUR CODE HERE
        gen_string = list(prefix)
        gen_string = process_text_for_Ngram([gen_string], self.N)[0]
        for i in range(max_words):
          stringLen = len(gen_string)
          curr_prefix = "".join(gen_string[stringLen - self.N + 1:])
          if curr_prefix in self.counts:
            new_word = np.random.choice(self.all_words[curr_prefix], p = self.all_probs[curr_prefix])
            gen_string.append(new_word)
            if (new_word == ' <eos>'):
              break
          else:
            gen_string.append('<unk>')

        return gen_string

    # Extra utility functions that you think will be useful can go below
    # YOUR CODE HERE
    def get_word_prob(self, Ngram, nextWord):
      if (Ngram in self.counts):
        if nextWord in self.counts[Ngram]:
          return self.counts[Ngram][nextWord] / self.total_counts[Ngram]
        else:
          return 0
      else:
        return 0

    def get_next_word_probs(self, Ngram):
      if Ngram in self.counts:
        ret_dict = {}
        probs = self.all_probs[Ngram]
        for i, word in enumerate(self.all_words[Ngram]):
          ret_dict[word] = probs[i]
        return ret_dict
      else:
        return {}

    def ret_next_char(self, prefix):
      new_text = self.sample_text(prefix, 1)
      new_char = new_text[len(new_text) - 1]
      if new_char == '<sos> ' or new_char == ' <eos>' or new_char == '\n' or new_char == '\t' or new_char == "\'":
        new_char = ' '
      elif new_char == '<unk>':
        new_char = np.random.choice(['e', 'a', 't'], p=[0.34, 0.33, 0.33])
      return new_char

    def get_three_top(self, prefix):
      three_top = set()
      for i in range(15):
        three_top.add(self.ret_next_char(prefix))
        if len(three_top) == 3:
          break
      if len(three_top) < 3:
        three_top.add('e')
        if len(three_top) < 3:
          three_top.add('a')
          if len(three_top) < 3:
            three_top.add('t')
      return list(three_top)

class WordNGramLMWithInterpolation(WordNGramLM):
    """
    Remember you can use the inheritance from WordNGramLM in your implementation!
    """

    def __init__(self, N: int, lambdas):

        """
        Constructor for WordNGramLMWithInterpolation class.
        Inputs:
            - N: int, the N in N-gram
            - lambdas: List[float], the list of lambdas for interpolation between 1-gram, 2-gram, 3-gram, ..., N-gram models
                Note: The length of lambdas should be N. The sum of lambdas should be 1. lambdas[0] corresponds to 1-gram model, lambdas[1] corresponds to 2-gram model and so on.
        """

        # YOUR CODE HERE
        super().__init__(N)
        self.lambdas = lambdas
        self.wordNGrams = []
        for n in range(1, self.N + 1):
          newWordNGram = WordNGramLM(n)
          self.wordNGrams.append(newWordNGram)

    def fit(self, train_data):

        """
        Trains an N-gram language model with interpolation.

        Inputs:
            - train_data: str, sentences in the training data

        """

        # YOUR CODE HERE
        for n in range(1, self.N + 1):
          self.wordNGrams[n-1].fit(train_data)

    def sample_text(self, prefix: str = "<sos>", max_words: int = 100) -> str:

        """
        Samples text from the N-gram language model with interpolation.

        Inputs:
            - prefix: str, the prefix to start the sampling from. Can also be multiple words separated by spaces.
            - max_words: int, the maximum number of words to sample

        Outputs:
            - str, the sampled text

        Note: Please use np.random.choice for sampling next words
        """

        # YOUR CODE HERE
        gen_string = list(prefix)
        gen_string = process_text_for_Ngram([gen_string], self.N)[0]
        for i in range(max_words):
          stringLen = len(gen_string)
          word_probs = {}
          for n in range(1, self.N + 1):
            curr_prefix = "".join(gen_string[stringLen - n + 1:])
            new_word_probs = self.wordNGrams[n - 1].get_next_word_probs(curr_prefix)
            for new_word, prob in new_word_probs.items():
              if new_word not in word_probs:
                word_probs[new_word] = self.lambdas[n - 1] * prob
              else:
                word_probs[new_word] = word_probs[new_word] + self.lambdas[n - 1] * prob

          total_prob_sum = np.sum(np.asarray(list(word_probs.values())))
          if total_prob_sum > 0:
            for word, probability in word_probs.items():
              word_probs[word] = probability / total_prob_sum
            new_word = np.random.choice(list(word_probs.keys()), p=list(word_probs.values()))
            gen_string.append(new_word)
            if (new_word == ' <eos>'):
              break
          else:
            gen_string.append('<unk>')

        return gen_string
    
def add_eos(data):
    """
    Adds an <eos> token to the end of each line in the data.

    Inputs:
    - data: a list of strings where each string is a line of text

    Returns:
    - a list of strings where each string is a line of text with <eos> token appended
    """
    # YOUR CODE HERE
    eos_appended = []
    for letter in data:
      eos_appended.append(letter)
    eos_appended.append(' <eos>')
    return eos_appended

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self, model: WordNGramLMWithInterpolation):
       self.model = model

    @classmethod
    def load_training_data(cls, work_dir):
        # your code here
        # this particular model doesn't train
        df = pd.read_csv(os.path.join(work_dir, 'movie_data_no_blanks.csv'))
        sentences = list(df['sentence'].values)
        new_sentences = []
        for sentence in sentences:
            new_sentences.append(add_eos(list(sentence.lower())))
        return new_sentences

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        self.model = WordNGramLMWithInterpolation(10, [0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.17, 0.1, 0.25, 0.35])
        self.model.fit(data)

    def run_pred(self, data):
        # your code here
        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            # this model just predicts a random character each time
            top_guesses = self.model.get_three_top(inp)
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.pckl'), 'wb') as f:
            pickle.dump(self.model, f)
            f.close()

    @classmethod
    def load(cls, work_dir, train):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        if train:
           return MyModel(WordNGramLMWithInterpolation(10, [0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.1, 0.17, 0.25, 0.35]))
        with open(os.path.join(work_dir, 'model.pckl'), 'rb') as f:
           model = pickle.load(f)
           f.close()
        return MyModel(model)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel.load(args.work_dir, train=True)
        print('Loading training data')
        train_data = MyModel.load_training_data(args.work_dir)
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir, train=False)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
