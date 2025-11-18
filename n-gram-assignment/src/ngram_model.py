import random

class TrigramModel:
    def __init__(self):
        """
        Initializes the TrigramModel.
        """
        # TODO: Initialize any data structures you need to store the n-gram counts.
       
        pass

    def fit(self, text):
        """
        Trains the trigram model on the given text.

        Args:
            text (str): The text to train the model on.
        """
        # TODO: Implement the training logic.
        # This will involve:
        # 1. Cleaning the text (e.g., converting to lowercase, removing punctuation).
        # 2. Tokenizing the text into words.
        # 3. Padding the text with start and end tokens.
        # 4. Counting the trigrams.
        pass

    def generate(self, max_length=50):
        """
        Generates new text using the trained trigram model.

        Args:
            max_length (int): The maximum length of the generated text.

        Returns:
            str: The generated text.
        """
        # TODO: Implement the generation logic.
        # This will involve:
        # 1. Starting with the start tokens.
        # 2. Probabilistically choosing the next word based on the current context.
        # 3. Repeating until the end token is generated or the maximum length is reached.
        pass
