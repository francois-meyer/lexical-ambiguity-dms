import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
from word2dm import *
from density_matrices import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--corpus_path", default="../data/mock.txt")
parser.add_argument("--output_path", default="../../dms/word2dm.model")
parser.add_argument("--n_dim", type=int, default=17)
parser.add_argument("--m_dim", type=int, default=17)
parser.add_argument("--window_size", type=int, default=5)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--neg_samples", type=int, default=5)
parser.add_argument("--min_count", type=int, default=50)
parser.add_argument("--subsampling", type=float, default=1e-5)
parser.add_argument("--neg_table_size", type=float, default=10e8)

args = parser.parse_args()
corpus_path = args.corpus_path
output_path = args.output_path
n_dim = args.n_dim
m_dim = args.m_dim
window_size = args.window_size
num_epochs = args.num_epochs
lr = args.lr
batch_size = args.batch_size
neg_samples = args.neg_samples
min_count = args.min_count
subsampling = args.subsampling
neg_table_size = args.neg_table_size

vocab = Word2DMVocab(corpus_path=corpus_path, min_count=min_count, subsampling=subsampling, neg_table_size=neg_table_size)
vocab.build_vocab()
dataset = SkipGramDataset(corpus_path=corpus_path, vocab=vocab, window_size=window_size, neg_samples=neg_samples)
loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

# Initialise model
vocab_size = len(vocab.itos)
model = Word2DM(vocab_size, n_dim, m_dim)

if torch.cuda.device_count() > 1:
    print("Let's use %d GPUs!" % torch.cuda.device_count())
    model = nn.DataParallel(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)
optimizer = optim.SparseAdam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    epoch_loss = 0
    batch_num = 0
    for target_ids, context_ids, neg_ids in tqdm(loader):
        if target_ids is None:
            continue

        target_ids = target_ids.to(device)
        context_ids = context_ids.to(device)
        neg_ids = neg_ids.to(device)
        batch_size = len(target_ids)
        optimizer.zero_grad()
        loss = model(target_ids, context_ids, neg_ids, batch_size)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_num % 1000000 == 0:
            print("Batch %d loss %f" % (batch_num, loss.item()))
        batch_num += 1

    print("Epoch %d loss: %f" % (epoch+1, epoch_loss))

if torch.cuda.device_count() > 1:
    model = model.module

vocab.neg_table = None # free up memory
dm_array = model.get_density_matrices().cpu().numpy()
dms = DensityMatrices(vocab, dm_array)
print("Trained density matrices.")

dms.normalise()
print("Normalised density matrices.")

dms.save(output_path)
print("Saved normalised density matrices.")