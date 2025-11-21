import random
from collections import defaultdict

class TrigramModel:
    def __init__(self):
        self.trigrams = defaultdict(list)
        self.starts = []

    def fit(self, text: str):
        """Train the model on input text."""
        tokens = text.split()

        if len(tokens) < 3:
            return

        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            self.trigrams[(w1, w2)].append(w3)

            if i == 0:
                self.starts.append((w1, w2))

    def generate(self, max_words=50):
        """Generate text based on trigram transitions."""
        if not self.starts:
            return ""

        w1, w2 = random.choice(self.starts)
        output = [w1, w2]

        for _ in range(max_words - 2):
            key = (w1, w2)
            if key not in self.trigrams:
                break

            next_word = random.choice(self.trigrams[key])
            output.append(next_word)

            w1, w2 = w2, next_word

        return " ".join(output)
