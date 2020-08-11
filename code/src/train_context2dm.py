from itertools import groupby
import pickle
from nltk.corpus import stopwords
STOP_WORDS = stopwords.words("english")
STOP_WORDS.remove("few") # remove few since it's in the evaluation data

from utils import *
from context2dm import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--corpus_path", default="../data/mock.txt")
parser.add_argument("--output_path", default="../../dms/context2dm.model")
parser.add_argument("--embedding_path", default="../../pretrained/word2vec-17.pickled")
parser.add_argument("--eval_words_path", default="../notebooks/eval_words.pickle")
parser.add_argument("--cluster_alg", type=str, default="hac")
parser.add_argument("--cluster_eval", type=str, default="vrc")

args = parser.parse_args()
corpus_path = args.corpus_path
output_path = args.output_path
embedding_path = args.embedding_path
eval_words_path = args.eval_words_path
cluster_alg = args.cluster_alg
cluster_eval = args.cluster_eval

emb_dim = 17
dim = 17*17
embeddings = DensityMatrices.load(embedding_path)
window = 5


def tokenize(line):
    return line.split()

# Load data
words = pickle.load(open(eval_words_path, 'rb'))
eval_vocab = EvalVocab(words)

# Load data
eos_token = "<eos>"
TEXT = Field(sequential=True, tokenize=tokenize, use_vocab=True, eos_token=eos_token)
corpus = Corpus(path=corpus_path, text_field=TEXT, newline_eos=True, encoding="latin1")
TEXT.build_vocab(corpus, min_freq=1)
vocab = TEXT.vocab

# Iterate corpus
model = Context2DM(vocab=vocab)
count = 0
last_input_words = None
last_vocab_ids = None

for example in tqdm(corpus.iters(corpus, batch_size=1, bptt_len=10)): # batch_size must be 1

    # Encode sentence words with vocab and BERT ids
    vocab_ids = [index.item() for index in example.text.T[0]]
    input_words = [vocab.itos[index] for index in example.text.T[0]]

    # If the last sentence of the previous batch was incomplete, add it to the start of this batch
    if last_input_words is not None:
        input_words = last_input_words + input_words
        last_input_words = None

    # Separate the encoded sentences
    split_input_words = [list(j) for i, j in groupby(input_words, key=lambda x: x != eos_token) if i]

    # Check if last sentence in batch is complete - if not, remove from current batch and store for next batch
    if split_input_words[-1][-1] != eos_token:
        last_input_words = split_input_words.pop()

    # Process sentences as context windows
    for sentence in split_input_words:
        for i, center_word in enumerate(sentence):
            if center_word.isalpha() and center_word not in STOP_WORDS:
                context_words = sentence[max(0, i - window): min(len(sentence), i + window + 1)]
                context_words = [word for word in context_words if word != center_word and word in embeddings]

                context_sum = np.zeros(emb_dim)
                for context_word in context_words:
                    context_sum += embeddings[context_word]
                if context_sum.any():
                    context_centroid = context_sum / len(context_words)
                    model.add_word_rep(vocab.stoi[center_word], context_centroid)
    count += 1

# Process last sentence as context window
if last_input_words is not None:
    for i, center_word in enumerate(last_input_words):

        if center_word.isalpha() and center_word not in STOP_WORDS:
            context_words = last_input_words[max(0, i - window): min(len(last_input_words), i + window + 1)]
            context_words = [word for word in context_words if word != center_word and word in embeddings]

            context_sum = np.zeros(emb_dim)
            for context_word in context_words:
                context_sum += embeddings[context_word]
            if context_sum.any():
                context_centroid = context_sum / len(context_words)
                model.add_word_rep(vocab.stoi[center_word], context_centroid)  # eval_vocab.stoi


print("made it here")
model.cluster_word_reps(cluster_alg=cluster_alg, cluster_eval=cluster_eval)
print("1")
model.initialise_dms(dim=emb_dim)
model.build_dms()
print("2")
model.normalise()
print("3")
dms = DensityMatrices(vocab, model.dm)
dms.save(output_path)

