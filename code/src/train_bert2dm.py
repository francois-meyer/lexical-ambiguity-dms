from itertools import groupby
from utils import *
from bert2dm import *
from density_matrices import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--corpus_path",default="../data/mock.txt")
parser.add_argument("--output_path", default="../../dms/bert2dm.model")
parser.add_argument("--dim", type=int, default=17)
parser.add_argument("--extract_layers", type=int, default=1)
parser.add_argument("--combine_layers", default="no")
parser.add_argument("--reduce_dim", type=bool, default=True)
parser.add_argument("--reduce_alg", type=str, default="pca")
parser.add_argument("--remove_stopwords", type=bool, default=True)
parser.add_argument("--cluster_alg", type=str, default="kmeans")
parser.add_argument("--cluster_eval", type=str, default="sc")

args = parser.parse_args()
corpus_path = args.corpus_path
output_path = args.output_path
dim = args.dim
extract_layers = args.extract_layers
combine_layers = args.combine_layers
reduce_dim = args.reduce_dim
reduce_alg = args.reduce_alg
remove_stopwords = args.remove_stopwords
cluster_alg = args.cluster_alg
cluster_eval = args.cluster_eval

# Load BERT and BERT tokenizer
model_class = BertModel
tokenizer_class = BertTokenizer
pretrained_weights = 'bert-base-uncased'
wp_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
config = BertConfig.from_pretrained(pretrained_weights, output_hidden_states=True)
bert = model_class.from_pretrained(pretrained_weights, config=config)


def tokenize_words(sentence):
    # Tokenize sentence into words (using the word separation of a wordpiece tokenizer)
    word_pieces = wp_tokenizer.tokenize(sentence)
    words_separated = wp_tokenizer.convert_tokens_to_string(word_pieces)
    words = words_separated.split()
    return words


# Load data
TEXT = Field(sequential=True, tokenize=tokenize_words, use_vocab=True, eos_token="<eos>")
corpus = Corpus(path=corpus_path, text_field=TEXT, newline_eos=True)
TEXT.build_vocab(corpus, min_freq=1)
vocab = TEXT.vocab
eos_vocab_id = vocab.stoi[TEXT.eos_token]

# Create model and train
model = Bert2DM(vocab=TEXT.vocab, corpus_reps=reduce_dim)

last_vocab_ids = None
for example in tqdm(corpus.iters(corpus, batch_size=1, bptt_len=1000)): # batch_size must be 1
    vocab_ids = [index.item() for index in example.text.T[0]]

    # If the last sentence of the previous batch was incomplete, add it to the start of this batch
    if last_vocab_ids is not None:
        vocab_ids = last_vocab_ids + vocab_ids
        last_vocab_ids = None

    # Separate the encoded sentences
    split_vocab_ids = [list(j) for i, j in groupby(vocab_ids, key=lambda x: x != eos_vocab_id) if i]

    # Check if last sentence in batch is complete - if not, remove from current batch and store for next batch
    if split_vocab_ids[-1][-1] != eos_vocab_id:
        last_vocab_ids = split_vocab_ids.pop()

    # Convert vocab ids back to words so BERT can process sentence with words
    split_vocab_tokens = [[vocab.itos[vocab_id] for vocab_id in vocab_ids] for vocab_ids in split_vocab_ids]

    # Process sentences with BERT
    with torch.no_grad():
        for i, vocab_tokens in enumerate(split_vocab_tokens):
            model.bert_process_sentence(vocab_tokens, split_vocab_ids[i], bert, wp_tokenizer, args)

# Process last sentence with BERT
if last_vocab_ids is not None:
    vocab_tokens = [vocab.itos[vocab_id] for vocab_id in last_vocab_ids]
    with torch.no_grad():
        model.bert_process_sentence(vocab_tokens, last_vocab_ids, bert, wp_tokenizer, args)

print("made it here")
model.cluster_word_reps(cluster_alg=cluster_alg, cluster_eval=cluster_eval)
print('1')
if reduce_dim:
    model.reduce_word_reps(dim=dim, reduce_alg=reduce_alg)
    print('2')
    model.build_dms()
    print('3')
model.normalise()
print('4')
dms = DensityMatrices(TEXT.vocab, model.dm)
dms.save(output_path)

