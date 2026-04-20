import math, os, pickle, re
from typing import Tuple, List, Dict


class BayesClassifier:
    """A simple BayesClassifier implementation"""

    def __init__(self):
        self.pos_freqs: Dict[str, int] = {}
        self.neg_freqs: Dict[str, int] = {}
        self.pos_filename: str = "pos.dat"
        self.neg_filename: str = "neg.dat"
        self.training_data_directory: str = "movie_reviews/"
        self.neg_file_prefix: str = "movies-1"
        self.pos_file_prefix: str = "movies-5"

        if os.path.isfile(self.pos_filename) and os.path.isfile(self.neg_filename):
            print("Data files found - loading to use cached values...")
            self.pos_freqs = self.load_dict(self.pos_filename)
            self.neg_freqs = self.load_dict(self.neg_filename)
        else:
            print("Data files not found - running training...")
            self.train()

    def train(self) -> None:
        _, __, files = next(os.walk(self.training_data_directory), (None, None, []))
        if not files:
            raise RuntimeError(f"Couldn't find path {self.training_data_directory}")

        for index, filename in enumerate(files, 1):
            print(f"Training on file {index} of {len(files)}: {filename}")

            text = self.load_file(os.path.join(self.training_data_directory, filename))
            tokens = self.tokenize(text)

            if filename.startswith(self.pos_file_prefix):
                self.update_dict(tokens, self.pos_freqs)
            elif filename.startswith(self.neg_file_prefix):
                self.update_dict(tokens, self.neg_freqs)

        self.save_dict(self.pos_freqs, self.pos_filename)
        self.save_dict(self.neg_freqs, self.neg_filename)

    def classify(self, text: str) -> str:
        tokens = self.tokenize(text)

        pos_prob = 0.0
        neg_prob = 0.0

        pos_total = sum(self.pos_freqs.values())
        neg_total = sum(self.neg_freqs.values())

        vocab = set(list(self.pos_freqs.keys()) + list(self.neg_freqs.keys()))
        V = len(vocab)

        for t in tokens:
            pos_count = self.pos_freqs.get(t, 0)
            neg_count = self.neg_freqs.get(t, 0)

            pos_prob += math.log((pos_count + 1) / (pos_total + V))
            neg_prob += math.log((neg_count + 1) / (neg_total + V))

        return "positive" if pos_prob > neg_prob else "negative"

    def load_file(self, filepath: str) -> str:
        with open(filepath, "r", encoding='utf8') as f:
            return f.read()

    def save_dict(self, dict: Dict, filepath: str) -> None:
        print(f"Dictionary saved to file: {filepath}")
        with open(filepath, "wb") as f:
            pickle.Pickler(f).dump(dict)

    def load_dict(self, filepath: str) -> Dict:
        print(f"Loading dictionary from file: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.Unpickler(f).load()

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        token = ""
        for c in text:
            if (
                re.match("[a-zA-Z0-9]", str(c)) != None
                or c == "'"
                or c == "_"
                or c == "-"
            ):
                token += c
            else:
                if token != "":
                    tokens.append(token.lower())
                    token = ""
                if c.strip() != "":
                    tokens.append(str(c.strip()))

        if token != "":
            tokens.append(token.lower())
        return tokens

    def update_dict(self, words: List[str], freqs: Dict[str, int]) -> None:
        for w in words:
            freqs[w] = freqs.get(w, 0) + 1

if __name__ == "__main__":
    # uncomment the below lines once you've implemented `train` & `classify`
    b = BayesClassifier()
    a_list_of_words = ["I", "really", "like", "this", "movie", ".", "I", "hope", "you", "like", "it", "too"]
    a_dictionary = {}
    b.update_dict(a_list_of_words, a_dictionary)
    assert a_dictionary["I"] == 2, "update_dict test 1"
    assert a_dictionary["like"] == 2, "update_dict test 2"
    assert a_dictionary["really"] == 1, "update_dict test 3"
    assert a_dictionary["too"] == 1, "update_dict test 4"
    print("update_dict tests passed.")

    pos_denominator = sum(b.pos_freqs.values())
    neg_denominator = sum(b.neg_freqs.values())

    print("\nThese are the sums of values in the positive and negative dicitionaries.")
    print(f"sum of positive word counts is: {pos_denominator}")
    print(f"sum of negative word counts is: {neg_denominator}")

    print("\nHere are some sample word counts in the positive and negative dicitionaries.")
    print(f"count for the word 'love' in positive dictionary {b.pos_freqs['love']}")
    print(f"count for the word 'love' in negative dictionary {b.neg_freqs['love']}")
    print(f"count for the word 'terrible' in positive dictionary {b.pos_freqs['terrible']}")
    print(f"count for the word 'terrible' in negative dictionary {b.neg_freqs['terrible']}")
    print(f"count for the word 'computer' in positive dictionary {b.pos_freqs['computer']}")
    print(f"count for the word 'computer' in negative dictionary {b.neg_freqs['computer']}")
    print(f"count for the word 'science' in positive dictionary {b.pos_freqs['science']}")
    print(f"count for the word 'science' in negative dictionary {b.neg_freqs['science']}")
    print(f"count for the word 'i' in positive dictionary {b.pos_freqs['i']}")
    print(f"count for the word 'i' in negative dictionary {b.neg_freqs['i']}")
    print(f"count for the word 'is' in positive dictionary {b.pos_freqs['is']}")
    print(f"count for the word 'is' in negative dictionary {b.neg_freqs['is']}")
    print(f"count for the word 'the' in positive dictionary {b.pos_freqs['the']}")
    print(f"count for the word 'the' in negative dictionary {b.neg_freqs['the']}")

    print("\nHere are some sample probabilities.")
    print(f"P('love'| pos) {(b.pos_freqs['love']+1)/pos_denominator}")
    print(f"P('love'| neg) {(b.neg_freqs['love']+1)/neg_denominator}")
    print(f"P('terrible'| pos) {(b.pos_freqs['terrible']+1)/pos_denominator}")
    print(f"P('terrible'| neg) {(b.neg_freqs['terrible']+1)/neg_denominator}")

    # # uncomment the below lines once you've implemented `classify`
    print("\nThe following should all be positive.")
    print(b.classify('I love computer science'))
    print(b.classify('this movie is fantastic'))
    print("\nThe following should all be negative.")
    print(b.classify('rainy days are the worst'))
    print(b.classify('computer science is terrible'))