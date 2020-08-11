from torchtext.datasets import LanguageModelingDataset
from torchtext.data import Field, BPTTIterator, Dataset
from density_matrices import DensityMatrices
import pandas as pd
import numpy as np
import torch
from collections import Counter
import math
from torchtext.datasets import WikiText2
from transformers import *
from tqdm import tqdm
import random


class Corpus(LanguageModelingDataset):

    @classmethod
    def iters(cls, dataset, batch_size=2, bptt_len=20, device=0, **kwargs):
        """Create iterator objects for splits of the WikiText-2 dataset.
        This is the simplest way to use the dataset, and assumes common
        defaults for field, vocabulary, and iterator parameters.
        Arguments:
            batch_size: Batch size.
            bptt_len: Length of sequences for backpropagation through time.
            device: Device to create batches on. Use -1 for CPU and None for
                the currently active GPU device.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose wikitext-2
                subdirectory the data files will be stored.
            wv_dir, wv_type, wv_dim: Passed to the Vocab constructor for the
                text field. The word vectors are accessible as
                train.dataset.fields['text'].vocab.vectors.
            Remaining keyword arguments: Passed to the splits method.
        """
        return BPTTIterator(dataset, batch_size=batch_size, bptt_len=bptt_len, device=device)


class EvalVocab:
    def __init__(self, words):
        self.itos = words
        self.stoi = {word: index for index, word in enumerate(words)}


class SkipGramVocab:

    def __init__(self, corpus_path, min_count, subsampling, neg_table_size):
        self.corpus_path = corpus_path
        self.min_count = min_count
        self.subsampling = subsampling
        self.neg_table_size = neg_table_size

        self.num_sentences = 0
        self.train_count = 0
        self.counter = Counter()
        self.stoi = {}
        self.itos = []
        self.subsampling_probs = {}
        self.neg_dist = None
        self.neg_table = []

    def build_vocab(self):
        self.count_words()
        self.apply_min_count()
        self.setup_negsampling()
        self.setup_subsampling()

    def count_words(self):
        with open(self.corpus_path, encoding="latin1") as file:
            for line in file:
                tokens = line.split()
                if len(tokens) > 0:
                    self.num_sentences += 1
                for token in tokens:
                    self.counter.update([token])
        print("Raw vocab size is %d." % len(self.counter))

    def apply_min_count(self):
        discard_count = 0
        raw_train_count = 0
        for word in list(self.counter):
            raw_train_count += self.counter[word]
            if self.counter[word] < self.min_count:
                del self.counter[word]
                discard_count += 1
            else:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)
                self.train_count += self.counter[word]

        print("Discarded %d words with count less than %d." % (discard_count, self.min_count))
        print("Final vocab size is %d." % len(self.itos))
        print("Training words down from %d to %d." % (raw_train_count, self.train_count))

    def setup_negsampling(self):
        # Compute distribution for negative sampling
        self.neg_dist = np.array(list(self.counter.values()))
        self.neg_dist = np.power(self.neg_dist, 0.75)
        self.neg_dist = self.neg_dist / self.neg_dist.sum()

        self.neg_dist = np.round(self.neg_dist * self.neg_table_size)
        for word_index, count in enumerate(self.neg_dist):
            self.neg_table += [word_index] * int(count)
        self.neg_dist = None  # free up memory
        self.neg_table = np.array(self.neg_table)
        np.random.shuffle(self.neg_table)

    def setup_subsampling(self):
        subsampling_count = 0
        threshold_count = self.train_count * self.subsampling
        print("Subsampling count threshold is %d." % threshold_count)

        for word in self.itos:
            sample_prob = (math.sqrt(self.counter[word] / threshold_count) + 1) * (threshold_count / self.counter[word])
            self.subsampling_probs[word] = min(sample_prob, 1.0)
            if sample_prob < 1.0:
                subsampling_count += 1
        print("Subsampling %d words with frequency greater than %f." % (subsampling_count, self.subsampling))

    def contains(self, word):
        return word in self.stoi

    def get_index(self, word):
        return self.stoi[word]

    def size(self):
        return len(self.itos)


class Word2DMVocab:

    def __init__(self, corpus_path, min_count, subsampling, neg_table_size):
        self.corpus_path = corpus_path
        self.min_count = min_count
        self.subsampling = subsampling
        self.neg_table_size = neg_table_size

        self.num_sentences = 0
        self.train_count = 0
        self.counter = Counter()
        self.stoi = {}
        self.itos = []
        self.subsampling_probs = {}
        self.neg_dist = None
        self.neg_table = []

    def build_vocab(self):
        self.count_words()
        self.apply_min_count()
        self.setup_negsampling()
        self.setup_subsampling()

    def count_words(self):
        with open(self.corpus_path, encoding="latin1") as file:
            for line in file:
                tokens = line.split()
                if len(tokens) > 0:
                    self.num_sentences += 1
                for token in tokens:
                    self.counter.update([token])
        print("Raw vocab size is %d." % len(self.counter))

    def apply_min_count(self):
        discard_count = 0
        raw_train_count = 0
        for word in list(self.counter):
            raw_train_count += self.counter[word]
            if self.counter[word] < self.min_count:
                del self.counter[word]
                discard_count += 1
            else:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)
                self.train_count += self.counter[word]

        print("Discarded %d words with count less than %d." % (discard_count, self.min_count))
        print("Final vocab size is %d." % len(self.itos))
        print("Training words down from %d to %d." % (raw_train_count, self.train_count))

    def setup_negsampling(self):
        # Compute distribution for negative sampling
        self.neg_dist = np.array(list(self.counter.values()))
        self.neg_dist = np.power(self.neg_dist, 0.75)
        self.neg_dist = self.neg_dist / self.neg_dist.sum()

        self.neg_dist = np.round(self.neg_dist * self.neg_table_size)
        for word_index, count in enumerate(self.neg_dist):
            self.neg_table += [word_index] * int(count)
        self.neg_dist = None  # free up memory
        self.neg_table = np.array(self.neg_table)
        np.random.shuffle(self.neg_table)

    def setup_subsampling(self):
        subsampling_count = 0
        threshold_count = self.train_count * self.subsampling
        print("Subsampling count threshold is %d." % threshold_count)

        for word in self.itos:
            sample_prob = (math.sqrt(self.counter[word] / threshold_count) + 1) * (threshold_count / self.counter[word])
            self.subsampling_probs[word] = min(sample_prob, 1.0)
            if sample_prob < 1.0:
                subsampling_count += 1
        print("Subsampling %d words with frequency greater than %f." % (subsampling_count, self.subsampling))

    def contains(self, word):
        return word in self.stoi

    def get_index(self, word):
        return self.stoi[word]

    def size(self):
        return len(self.itos)


class SkipGramDataset(Dataset):

    def __init__(self, corpus_path, vocab, window_size, neg_samples):
        self.window_size = window_size
        self.corpus_file = open(corpus_path, encoding="latin1")
        self.vocab = vocab
        self.vocab_size = len(vocab.itos)
        self.corpus_size = vocab.num_sentences
        self.neg_samples = neg_samples
        self.neg_index = 0

    def __len__(self):
        return self.corpus_size

    def __getitem__(self, idx):
        while True:
            line = self.corpus_file.readline()
            if not line:
                self.corpus_file.seek(0, 0)
                line = self.corpus_file.readline()

            if len(line) > 1:
                words = line.split()
                if len(words) > 1:
                    word_ids = [self.vocab.stoi[word] for word in words
                                if word in self.vocab.stoi and random.random() < self.vocab.subsampling_probs[word]]
                    return self.generate_sgns_predictions(word_ids)

    def generate_sgns_predictions(self, word_ids):
        if len(word_ids) > 100 or len(word_ids) <= 1:
            return [], [], torch.LongTensor()

        dynamic_window_size = int(random.random() * self.window_size) + 1
        target_ids = []
        context_ids = []
        for i, target_id in enumerate(word_ids):
            window_ids = word_ids[max(0, i - dynamic_window_size): min(len(word_ids), i + dynamic_window_size + 1)]
            window_ids = [word for word in window_ids if word != target_id]
            target_ids.extend([target_id] * len(window_ids))
            context_ids.extend(window_ids)

        if len(target_ids) == 0:
            return [], [], torch.LongTensor()

        total_neg_samples = len(target_ids) * self.neg_samples
        neg_ids = self.vocab.neg_table[self.neg_index: self.neg_index + total_neg_samples]
        self.neg_index = (self.neg_index + total_neg_samples) % len(self.vocab.neg_table)

        if len(neg_ids) != total_neg_samples:
            neg_ids = np.concatenate((neg_ids, self.vocab.neg_table[0: self.neg_index]))
        neg_ids = torch.from_numpy(neg_ids).view(len(target_ids), -1)

        return target_ids, context_ids, neg_ids

    def collate_fn(self, batch_list):
        target_ids = [target_id for batch in batch_list for target_id in batch[0]]
        context_ids = [context_id for batch in batch_list for context_id in batch[1]]
        neg_ids = [neg_samples for batch in batch_list for neg_samples in batch[2]]

        if len(target_ids) == 0:
            return None, None, None

        target_ids = torch.LongTensor(target_ids)
        context_ids = torch.LongTensor(context_ids)
        neg_ids = torch.stack(neg_ids)

        return target_ids, context_ids, neg_ids


class CBOWDataset(Dataset):

    def __init__(self, corpus_path, vocab, window_size, neg_samples):
        self.window_size = window_size
        self.corpus_file = open(corpus_path, encoding="latin1")
        self.vocab = vocab
        self.vocab_size = len(vocab.itos)
        self.corpus_size = vocab.num_sentences
        self.neg_samples = neg_samples
        self.neg_index = 0
        self.name = "CBOWDataset"

    def __len__(self):
        return self.corpus_size

    def __getitem__(self, idx):
        while True:
            line = self.corpus_file.readline()
            if not line:
                self.corpus_file.seek(0, 0)
                line = self.corpus_file.readline()

            if len(line) > 1:
                words = line.split()
                if len(words) > 1:
                    word_ids = [self.vocab.stoi[word] for word in words
                                if word in self.vocab.stoi and random.random() < self.vocab.subsampling_probs[word]]
                    return self.generate_cbowns_predictions(word_ids)

    def generate_cbowns_predictions(self, word_ids):
        if len(word_ids) > 100 or len(word_ids) <= 1:
            return [], [], torch.LongTensor(), []

        dynamic_window_size = int(random.random() * self.window_size) + 1
        target_ids = []
        context_ids = []
        for i, target_id in enumerate(word_ids):
            window_ids = word_ids[max(0, i - dynamic_window_size): min(len(word_ids), i + dynamic_window_size + 1)]
            window_ids = [word for word in window_ids if word != target_id]

            if len(window_ids) > 0:
                target_ids.append(target_id)
                context_ids.append(window_ids)

        if len(target_ids) == 0:
            return [], [], torch.LongTensor(), []

        # Pad context ids
        context_lens = [len(context) for context in context_ids]
        context_ids = [context + [self.vocab_size] * (self.window_size * 2 - len(context)) for context in context_ids]
        context_ids = [torch.LongTensor(context) for context in context_ids]

        # Negative sampling
        total_neg_samples = len(target_ids) * self.neg_samples
        neg_ids = self.vocab.neg_table[self.neg_index: self.neg_index + total_neg_samples]
        self.neg_index = (self.neg_index + total_neg_samples) % len(self.vocab.neg_table)
        if len(neg_ids) != total_neg_samples:
            neg_ids = np.concatenate((neg_ids, self.vocab.neg_table[0: self.neg_index]))
        neg_ids = torch.from_numpy(neg_ids).view(len(target_ids), -1)

        return target_ids, context_ids, neg_ids, context_lens

    def collate_fn(self, batch_list):
        target_ids = [target_id for batch in batch_list for target_id in batch[0]]
        context_ids = [context_id for batch in batch_list for context_id in batch[1]]
        neg_ids = [neg_samples for batch in batch_list for neg_samples in batch[2]]
        context_lens = [context_len for batch in batch_list for context_len in batch[3]]

        if len(target_ids) == 0:
            return None, None, None, None

        target_ids = torch.LongTensor(target_ids)
        context_ids = torch.stack(context_ids)
        neg_ids = torch.stack(neg_ids)
        context_lens = torch.LongTensor(context_lens)

        return target_ids, context_ids, neg_ids, context_lens


def main():
    print("python main function")
    path = "../data/mock.txt"
    path = "/home/frmeyer/uva/project/data/brown_preprocessed.txt"
    tokenize = lambda x: x.split()
    TEXT = Field(sequential=True, tokenize=tokenize, lower=True)

    print("Initialising corpus dataset")
    corpus = Corpus(path=path, text_field=TEXT)

    print("Building field vocab")
    TEXT.build_vocab(corpus, vectors=None)

    for item in iter(corpus):
        print("Test")
        print(item)


def main_wiki2():

    # Load BERT
    model_class = BertModel
    tokenizer_class = BertTokenizer
    pretrained_weights = 'bert-base-uncased'
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    bert = model_class.from_pretrained(pretrained_weights)

    # Load data
    TEXT = Field(sequential=True, tokenize=tokenizer.tokenize, use_vocab=True, eos_token="<eos>")
    train, valid, test = WikiText2.splits(TEXT)
    TEXT.build_vocab(train, min_freq=1)

    # Map vocab ids to BERT ids
    itobert = tokenizer.convert_tokens_to_ids(TEXT.vocab.itos)
    eos_vocab_index = TEXT.vocab.stoi[TEXT.eos_token]
    itobert[eos_vocab_index] = tokenizer.sep_token_id

    train_iter, valid_iter, test_iter = BPTTIterator.splits((train, valid, test), batch_size=32, bptt_len=30, device=0,
                                                                repeat=False)

    for batch in tqdm(train_iter):
        text, targets = batch.text, batch.target
        print(text.shape)
        for sentence in text:
            input_ids = [itobert[int(index)] for index in sentence]
            input_words = [tokenizer.ids_to_tokens[input_id] for input_id in input_ids]
            print(input_words)


def assess_dataset():
    DROP_COLS = {"gs2011": ["participant", "input"],
                 "ks2013": ["annotator", "score"]}
    SEQ_COLS = {"gs2011": [["subject", "verb", "object"], ["subject", "landmark", "object"]],
                "ks2013": [["verb1", "object1"], ["verb2", "object2"]]}
    LOHI_COL = {"gs2011": "hilo",
                "ks2013": None}
    ANNOTATOR_COL = {"gs2011": "participant",
                     "ks2013": "annotator"}
    SCORE_COL = {"gs2011": "input",
                 "ks2013": "score"}

    model30 = DensityMatrices.load("../../dms/wiki2-30.model")

    eval_path = "../evaluation"
    dataset_paths = [eval_path + "/GS2011data.txt", eval_path + "/KS2013-EMNLP.txt"]
    dataset_names = ["gs2011", "ks2013"]

    for i, dataset_path in enumerate(dataset_paths):
        total = 0
        missing = 0

        dataset_name = dataset_names[i]
        df = pd.read_csv(dataset_path, delimiter=" ")
        examples_df = df.drop(DROP_COLS[dataset_name], axis=1).drop_duplicates().reset_index(drop=True)
        for index, row in examples_df.iterrows():
            total += 1
            sequence1 = [row[col] for col in SEQ_COLS[dataset_name][0]]
            sequence2 = [row[col] for col in SEQ_COLS[dataset_name][1]]
            for word in sequence1 + sequence2:
                if not model30.contains(word):
                    missing += 1
                    examples_df.drop(index, inplace=True)
                    break
        examples_df.reset_index(inplace=True, drop=True)
        print(examples_df)
        print("%s is missing %d out of %d" % (dataset_name, missing, total))


if __name__ == '__main__':
    main()



