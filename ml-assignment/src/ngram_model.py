import random
import math
from collections import defaultdict


class TrigramModel:
    def __init__(self):
        self.trigram_counts = defaultdict(lambda: defaultdict(int))
        self.bigram_counts = defaultdict(int)
        self.vocab = set()

    def train(self, text):
        """
        Train the trigram model on list of tokens.
        """
        for i in range(len(text) - 2):
            w1, w2, w3 = text[i], text[i+1], text[i+2]
            self.trigram_counts[(w1, w2)][w3] += 1
            self.bigram_counts[(w1, w2)] += 1
            self.vocab.update([w1, w2, w3])

    def get_trigram_prob(self, w1, w2, w3):
        """
        Return probability of w3 given w1, w2.
        Add-1 smoothing is used.
        """
        vocab_size = len(self.vocab)
        trigram_count = self.trigram_counts[(w1, w2)][w3]
        bigram_count = self.bigram_counts[(w1, w2)]

        return (trigram_count + 1) / (bigram_count + vocab_size)

    def generate_next_word(self, w1, w2):
        """
        Generate the next word based on trigram distribution.
        """
        candidates = self.trigram_counts[(w1, w2)]
        if not candidates:
            return random.choice(list(self.vocab))

        words = []
        probs = []

        vocab_size = len(self.vocab)
        total = self.bigram_counts[(w1, w2)] + vocab_size

        for word in self.vocab:
            count = candidates[word]
            words.append(word)
            probs.append((count + 1) / total)

        return random.choices(words, probs)[0]

    def perplexity(self, text):
        """
        Compute perplexity over a list of tokens.
        """
        log_prob_sum = 0
        N = len(text) - 2
        vocab_size = len(self.vocab)

        for i in range(N):
            w1, w2, w3 = text[i], text[i+1], text[i+2]
            trigram_count = self.trigram_counts[(w1, w2)][w3]
            bigram_count = self.bigram_counts[(w1, w2)]

            prob = (trigram_count + 1) / (bigram_count + vocab_size)
            log_prob_sum += math.log(prob)

        return math.exp(-log_prob_sum / N)
