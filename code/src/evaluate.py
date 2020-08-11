from scipy import stats
import pickle
import numpy as np
import pandas as pd
import torch
from density_matrices import DensityMatrices
from transformers import *
from sklearn.metrics.pairwise import cosine_similarity
from models import InferSent
import argparse

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199

# Command line arguments
NUM_SAMPLES = 100

EMBEDDING_FILENAMES = {"word2vec": "word2vec.pickled",
                       "glove": "glove.pickled",
                       "fasttext": "fasttext.pickled"}

TENSOR_FILENAMES = {"word2vec": "word2vec_tensors.pickled",
                    "glove": "glove_tensors.pickled",
                    "fasttext": "fasttext_tensors.pickled"}

EVAL_FILENAMES = {"gs2011": "GS2011data.txt",
                  "gs2012": "GS2012data.txt",
                  "ks2013-conll": "KS2013-CoNLL.txt",
                  "ks2013-emnlp": "KS2013-EMNLP.txt",
                  "ml2008": "ML2008.txt"}

SENTENCES_FILENAMES = {"gs2011": "GS2011_sentences.csv",
                       "gs2012": "GS2012_sentences.csv",
                       "ks2013-conll": "KS2013-CoNLL_sentences.csv",
                       "ks2013-emnlp": "KS2013-EMNLP_sentences.csv",
                       "ml2008": "ML2008_sentences.csv"}

DROP_COLS = {"gs2011": ["participant", "input"],
             "gs2012": ["annotator_id", "annotator_score"],
             "ks2013-conll": ["annotator_id", "annotator_score"],
             "ks2013-emnlp": ["annotator", "score"],
             "ml2008": ["participant", "input"]}
SEQ_COLS = {"gs2011": [["subject", "verb", "object"], ["subject", "landmark", "object"]],
            "gs2012": [["adj_subj", "subj", "verb", "adj_obj", "obj"], ["adj_subj", "subj", "landmark", "adj_obj", "obj"]],
            "ks2013-conll": [["adj_subj", "subj", "verb", "adj_obj", "obj"], ["adj_subj", "subj", "landmark", "adj_obj", "obj"]],
            "ks2013-emnlp": [["verb1", "object1"], ["verb2", "object2"]],
            "ml2008": [["noun", "verb"], ["noun", "landmark"]]}
LOHI_COL = {"gs2011": "hilo",
            "gs2012": None,
            "ks2013-conll": None,
            "ks2013-emnlp": None,
            "ml2008": "hilo"}
ANNOTATOR_COL = {"gs2011": "participant",
                 "gs2012": "annotator_id",
                 "ks2013-conll": "annotator_id",
                 "ks2013-emnlp": "annotator",
                 "ml2008": "participant"}
SCORE_COL = {"gs2011": "input",
             "gs2012": "annotator_score",
             "ks2013-conll": "annotator_score",
             "ks2013-emnlp": "score",
             "ml2008": "input"}
POS_TAGS = {"gs2011": ["noun", "verb", "noun"],
            "gs2012": ["adj", "noun", "verb", "adj", "noun"],
            "ks2013-conll": ["adj", "noun", "verb", "adj", "noun"],
            "ks2013-emnlp": ["verb", "noun"],
            "ml2008": ["noun", "verb"]}

eval_bert = True
eval_infersent = True
def evaluate(models, names, args, dataset_name, dataset_path, params):

    pretrained_path = args.pretrained_path
    dm_comps = params["dm_comps"]
    dm_methods = params["dm_methods"]
    vector_embeddings = params["vector_embeddings"]
    vector_comps = params["vector_comps"]
    vector_methods = params["vector_methods"]
    pickle_embedding_files = params["pickle_embedding_files"]
    sentences_path = params["sentences_path"]
    all_methods = dm_methods + vector_methods #+ ["bert"]

    pickle_embedding_paths = [pretrained_path + "/" + pickle_file for pickle_file in pickle_embedding_files]
    pretrained_embeddings = [load_embeddings_from_pickle(path) for path in pickle_embedding_paths]

    # Sentence encoding models
    other_models = []
    if eval_bert:
        other_models.append("bert")
    if eval_infersent:
        other_models.append("infersent1")
        other_models.append("infersent2")
    results = {method: [] for method in vector_methods + other_models + dm_methods}
    result_cols = []

    # Load dataset examples and discard missing ones
    df = pd.read_csv(dataset_path, delimiter=" ")
    examples_df = df.drop(DROP_COLS[dataset_name], axis=1).drop_duplicates().reset_index(drop=True)
    examples_df = discard_missing(examples_df, pretrained_embeddings, models, dataset_name)

    # Load mapping from word sequences to complete sentences
    complete_sentences_df = pd.read_csv(sentences_path + "/" + SENTENCES_FILENAMES[dataset_name])
    complete_sentences = dict(zip(complete_sentences_df["original"], complete_sentences_df["completed"]))

    # Initialise other compositional models
    # BERT
    model_class = BertModel
    tokenizer_class = BertTokenizer
    pretrained_weights = 'bert-base-uncased'
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    bert = model_class.from_pretrained(pretrained_weights)

    # InferSent
    complete_sentences_corpus = list(complete_sentences.values())

    # InferSent 1
    infersent1_vectors_path = "../../pretrained/infersent/GloVe/glove.840B.300d.txt"
    infersent1_model_path = "../../pretrained/infersent/infersent1.pkl"
    params_infersent1 = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048, 'pool_type': 'max', 'dpout_model': 0.0,
                         'version': 1}
    infersent1 = InferSent(params_infersent1)
    infersent1.load_state_dict(torch.load(infersent1_model_path))
    infersent1.set_w2v_path(infersent1_vectors_path)
    infersent1.build_vocab(complete_sentences_corpus, tokenize=True)

    # InferSent 2
    infersent2_vectors_path = "../../pretrained/infersent/fastText/crawl-300d-2M.vec"
    infersent2_model_path = "../../pretrained/infersent/infersent2.pkl"
    params_infersent2 = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048, 'pool_type': 'max', 'dpout_model': 0.0,
                         'version': 2}
    infersent2 = InferSent(params_infersent2)
    infersent2.load_state_dict(torch.load(infersent2_model_path))
    infersent2.set_w2v_path(infersent2_vectors_path)
    infersent2.build_vocab(complete_sentences_corpus, tokenize=True)

    # Compose density matrices of all dataset phrases and compute similarity scores
    dm_scores = {method: [] for method in dm_methods}
    vector_scores = {method: [] for method in vector_methods}
    bert_scores = []
    infersent1_scores = []
    infersent2_scores = []
    for index, row in examples_df.iterrows():
        sequence1 = [row[col] for col in SEQ_COLS[dataset_name][0]]
        sequence2 = [row[col] for col in SEQ_COLS[dataset_name][1]]
        for i, model in enumerate(models):
            sims = model.sequence_similarity(sequence1, sequence2, dm_comps, POS_TAGS[dataset_name])
            for method in dm_comps:
                dm_method = names[i] + "_" + method
                dm_scores[dm_method].append(sims[method])
        if eval_bert:
            bert_scores.append(get_bert_score(bert, tokenizer, sequence1, sequence2, complete_sentences)) # This takes time, could maybe be made more efficient
        if eval_infersent:
            infersent1_scores.append(get_infersent_score(infersent1, sequence1, sequence2, complete_sentences))
            infersent2_scores.append(get_infersent_score(infersent2, sequence1, sequence2, complete_sentences))

        # Vector models
        sims = get_vector_scores(vector_embeddings, vector_comps, pretrained_embeddings, sequence1, sequence2, dataset_name)
        for method in vector_methods:
            vector_scores[method].append(sims[method])

    for method in dm_methods:
        examples_df[method] = dm_scores[method]
    for method in vector_methods:
        examples_df[method] = vector_scores[method]
    if eval_bert:
        examples_df["bert_score"] = bert_scores
    if eval_infersent:
        examples_df["infersent1_score"] = infersent1_scores
        examples_df["infersent2_score"] = infersent2_scores

    # Compute Spearman's rank correlation coefficient
    result_cols.append("spearman")
    groupby_cols = list(set(SEQ_COLS[dataset_name][0]) | set(SEQ_COLS[dataset_name][1]))
    if LOHI_COL[dataset_name] is not None:
        groupby_cols += [LOHI_COL[dataset_name]]

    # Compute annotator score averages
    ave_df = df.drop(ANNOTATOR_COL[dataset_name], axis=1).groupby(groupby_cols).mean().reset_index()
    ave_df = pd.merge(left=examples_df, right=ave_df, how="left", on=groupby_cols).reset_index(drop=True)

    # Bootstrap
    samples = {}
    for method in dm_methods + vector_methods:
        samples[method] = []
    if eval_bert:
        samples["bert"] = []
    if eval_bert:
        samples["infersent1"] = []
        samples["infersent2"] = []

    for seed in range(NUM_SAMPLES):
        sampled_df = ave_df.sample(n=len(ave_df.index), replace=True, random_state=seed).reset_index(drop=True)
        for method in dm_methods + vector_methods:
            samples[method].append(sampled_df[SCORE_COL[dataset_name]].corr(sampled_df[method], method="spearman"))
        if eval_bert:
            samples["bert"].append(sampled_df[SCORE_COL[dataset_name]].corr(sampled_df["bert_score"], method="spearman"))
        if eval_infersent:
            samples["infersent1"].append(sampled_df[SCORE_COL[dataset_name]].corr(sampled_df["infersent1_score"], method="spearman"))
            samples["infersent2"].append(sampled_df[SCORE_COL[dataset_name]].corr(sampled_df["infersent2_score"], method="spearman"))

    for method in dm_methods + vector_methods:
        results[method].append(np.average(samples[method]))
    if eval_bert:
        results["bert"].append(np.average(samples["bert"]))
    if eval_infersent:
        results["infersent1"].append(np.average(samples["infersent1"]))
        results["infersent2"].append(np.average(samples["infersent2"]))

    results_df = pd.DataFrame.from_dict(results, orient="index", columns=result_cols)
    results_df = results_df.round(3)

    ttest_df = paired_ttest(samples, all_methods)

    return results_df, ttest_df


def discard_missing(examples_df, pretrained_embeddings, models, dataset_name):
    # Throw away examples not in the density matrices or word embeddings/tensors
    total_before = len(examples_df.index)
    dm_missing = 0
    embeddings_missing = 0
    dm_nans = 0
    for index, row in examples_df.iterrows():
        sequence1 = [row[col] for col in SEQ_COLS[dataset_name][0]]
        sequence2 = [row[col] for col in SEQ_COLS[dataset_name][1]]

        nan_detected = False
        for word in sequence1 + sequence2:
            if word not in pretrained_embeddings[0]:
                examples_df.drop(index, inplace=True)
                embeddings_missing += 1
                break
            models_contain = [model.contains(word) for model in models]
            if not all(models_contain):
                examples_df.drop(index, inplace=True)
                dm_missing += 1
                print("%s not in DMs" % word)
                break
            for model in models:
                if np.isnan(model.get_dm(word)).any():
                    examples_df.drop(index, inplace=True)
                    dm_nans += 1
                    nan_detected = True
                    break
            if nan_detected:
                nan_detected = False
                break

    examples_df.reset_index(inplace=True, drop=True)
    total_after = len(examples_df.index)
    print("%s: discarded %d out of %d examples" % (dataset_name, total_before - total_after, total_before))
    print("%d were not in the embeddings and %d were not in the density matrices" % (embeddings_missing, dm_missing))
    print("%d were nans in the density matrices" % dm_nans)

    return examples_df


def paired_ttest(samples, all_methods):

    p_values = {}
    for method1 in all_methods:
        p_values[method1] = []
        for method2 in all_methods:
            p_values[method1].append(stats.ttest_rel(samples[method1], samples[method2])[1])
    df = pd.DataFrame.from_dict(p_values, orient="index", columns=all_methods).round(4)

    return df


def load_tensors_from_txt(tensor_path):
    tensors = {}
    for line in open(tensor_path, 'r'):
        split_line = line.split()
        word = split_line[0].lower()
        flattened = np.array([float(val) for val in split_line[1:]])
        size_sqrt = int(np.sqrt(flattened.size))
        matrix = np.reshape(flattened, newshape=(size_sqrt, size_sqrt))
        tensors[word] = matrix
    print("Done with %s" % tensor_path)
    return tensors


def load_embeddings_from_txt(embedding_path):
    embeddings = {}
    for line in open(embedding_path, 'r'):
        split_line = line.split()
        word = split_line[0].lower()
        vector = np.array([float(val) for val in split_line[1:]])
        embeddings[word] = vector
    print("Done with %s" % embedding_path)
    return embeddings


def load_embeddings_from_pickle(pickle_path):
    # This works for both embeddings and tensors
    return pickle.load(open(pickle_path, 'rb'))


def load_and_save_embeddings(embedding_path, output_path):
    embeddings = load_embeddings_from_txt(embedding_path)
    with open(output_path, "wb") as output_file:
        pickle.dump(embeddings, output_file, protocol=2)
    print("Successfully loaded and saved %s." % embedding_path)


def load_and_save_tensors(tensor_path, output_path):
    tensors = load_tensors_from_txt(tensor_path)
    with open(output_path, "wb") as output_file:
        pickle.dump(tensors, output_file, protocol=2)
    print("Successfully loaded and saved %s." % tensor_path)


def get_infersent_score(infersent, sequence1, sequence2, complete_sentences):

    # Convert word sequences to complete sentences for BERT
    sent1 = complete_sentences[" ".join(sequence1)]
    sent2 = complete_sentences[" ".join(sequence2)]

    # Encode text
    seq_vectors = infersent.encode([sent1, sent2], tokenize=True)
    seq1_vector = seq_vectors[0]
    seq2_vector = seq_vectors[1]

    score = cosine_similarity([seq1_vector], [seq2_vector])[0, 0]
    return score


def get_bert_score(bert, tokenizer, sequence1, sequence2, complete_sentences):

    # Convert word sequences to complete sentences for BERT
    sent1 = complete_sentences[" ".join(sequence1)]
    sent2 = complete_sentences[" ".join(sequence2)]

    # Encode text
    seq1_ids = torch.tensor([tokenizer.encode(sent1, add_special_tokens=True)])
    seq2_ids = torch.tensor([tokenizer.encode(sent2, add_special_tokens=True)])

    with torch.no_grad():
        last_hidden_states1 = bert(seq1_ids)[0]
        last_hidden_states2 = bert(seq2_ids)[0]

    seq1_vector = last_hidden_states1[0, 0].numpy()
    seq2_vector = last_hidden_states2[0, 0].numpy()

    score = cosine_similarity([seq1_vector], [seq2_vector])[0, 0]
    return score


def get_baseline_score(embeddings, method):

    if method == "vector":
        pass
    elif method == "tensor":
        pass


def get_vector_scores(vector_embeddings, vector_comps, pretrained_embeddings, sequence1, sequence2, dataset_name):

    sims = {}
    for i, embedding in enumerate(vector_embeddings):
        for comp in vector_comps:
            sims[embedding + "_" + comp] = compose_and_score(pretrained_embeddings[i], comp, sequence1, sequence2, dataset_name)
    return sims


def compose_and_score(embeddings, comp, sequence1, sequence2, dataset_name):

    if comp == "verb_vector" or comp == "verb_tensor":
        if len(sequence1) == 2:
            vector1 = embeddings[sequence1[1]]
            vector2 = embeddings[sequence2[1]]
        elif len(sequence1) == 3:
            vector1 = embeddings[sequence1[1]]
            vector2 = embeddings[sequence2[1]]
        elif len(sequence1) == 5:
            vector1 = embeddings[sequence1[2]]
            vector2 = embeddings[sequence2[2]]

        if comp == "verb_tensor":
            vector1 = np.outer(vector1, vector1).flatten()
            vector2 = np.outer(vector2, vector2).flatten()

    elif comp == "mult":
        vector1 = np.ones(shape=embeddings[sequence1[0]].shape)
        for token in sequence1:
            vector1 *= embeddings[token]
        vector2 = np.ones(shape=embeddings[sequence2[0]].shape)
        for token in sequence2:
            vector2 *= embeddings[token]

    elif comp == "add":
        vector1 = np.zeros(shape=embeddings[sequence1[0]].shape)
        for token in sequence1:
            vector1 += embeddings[token]
        vector2 = np.zeros(shape=embeddings[sequence2[0]].shape)
        for token in sequence2:
            vector2 += embeddings[token]

    elif comp == "tensor":
        if len(sequence1) == 2:
            verb1_vector = embeddings[sequence1[POS_TAGS[dataset_name].index("verb")]]
            verb1_tensor = np.outer(verb1_vector, verb1_vector)
            noun1_vector = embeddings[sequence1[POS_TAGS[dataset_name].index("noun")]]
            vector1 = verb1_tensor @ noun1_vector

            verb2_vector = embeddings[sequence2[POS_TAGS[dataset_name].index("verb")]]
            verb2_tensor = np.outer(verb2_vector, verb2_vector)
            noun2_vector = embeddings[sequence2[POS_TAGS[dataset_name].index("noun")]]
            vector2 = verb2_tensor @ noun2_vector
        elif len(sequence1) == 3:
            subj1_vector = embeddings[sequence1[0]]
            verb1_vector = embeddings[sequence1[1]]
            verb1_tensor = np.outer(verb1_vector, verb1_vector)
            obj1_vector = embeddings[sequence1[2]]
            sub_obj_matrix1 = np.outer(subj1_vector, obj1_vector)
            matrix1 = verb1_tensor * sub_obj_matrix1
            vector1 = matrix1.flatten()

            subj2_vector = embeddings[sequence2[0]]
            verb2_vector = embeddings[sequence2[1]]
            verb2_tensor = np.outer(verb2_vector, verb2_vector)
            obj2_vector = embeddings[sequence2[2]]
            sub_obj_matrix2 = np.outer(subj2_vector, obj2_vector)
            matrix2 = verb2_tensor * sub_obj_matrix2
            vector2 = matrix2.flatten()

        elif len(sequence1) == 5:
            # Compose adj-subj verb adj-obj for sequence 1
            subj_adj1_vector = embeddings[sequence1[0]]
            subj_adj1_tensor = np.outer(subj_adj1_vector, subj_adj1_vector)
            subj1_vector = embeddings[sequence1[1]]
            subj_np1_vector = subj_adj1_tensor @ subj1_vector

            verb1_vector = embeddings[sequence1[2]]
            verb1_tensor = np.outer(verb1_vector, verb1_vector)

            obj_adj1_vector = embeddings[sequence1[3]]
            obj_adj1_tensor = np.outer(obj_adj1_vector, obj_adj1_vector)
            obj1_vector = embeddings[sequence1[4]]
            obj_np1_vector = obj_adj1_tensor @ obj1_vector

            sub_obj_matrix1 = np.outer(subj_np1_vector, obj_np1_vector)
            matrix1 = verb1_tensor * sub_obj_matrix1
            vector1 = matrix1.flatten()

            # Compose adj-subj verb adj-obj for sequence 2
            subj_adj2_vector = embeddings[sequence2[0]]
            subj_adj2_tensor = np.outer(subj_adj2_vector, subj_adj2_vector)
            subj2_vector = embeddings[sequence2[1]]
            subj_np2_vector = subj_adj2_tensor @ subj2_vector

            verb2_vector = embeddings[sequence2[2]]
            verb2_tensor = np.outer(verb2_vector, verb2_vector)

            obj_adj2_vector = embeddings[sequence2[3]]
            obj_adj2_tensor = np.outer(obj_adj2_vector, obj_adj2_vector)
            obj2_vector = embeddings[sequence2[4]]
            obj_np2_vector = obj_adj2_tensor @ obj2_vector

            sub_obj_matrix2 = np.outer(subj_np2_vector, obj_np2_vector)
            matrix2 = verb2_tensor * sub_obj_matrix2
            vector2 = matrix2.flatten()

    score = cosine_similarity([vector1], [vector2])[0, 0]

    return score


def main(args):

    # Define models to compare with
    # Vector models
    vector_embeddings = ["word2vec" "glove", "fasttext"]
    vector_comps = ["verb_vector", "mult", "add", "tensor"]
    vector_methods = [embedding + "_" + method for method in vector_comps for embedding in vector_embeddings]
    pickle_embedding_files = [EMBEDDING_FILENAMES[embedding] for embedding in vector_embeddings]
    pickle_tensor_files = [TENSOR_FILENAMES[embedding] for embedding in vector_embeddings]

    # My models
    dm_path = args.dm_path

    dm_models = {}

    dm_models["Context2DM"] = DensityMatrices.load(dm_path + "/context2dm.model")
    dm_models["Word2DM"] = DensityMatrices.load(dm_path + "/word2dm.model")
    dm_models["MS-Word2DM"] = DensityMatrices.load(dm_path + "/ms-word2dm.model")

    dm_models["BERT2DM PCA"] = DensityMatrices.load(dm_path + "/b2d-wiki10/bert2dm-pca.model")
    dm_models["BERT2DM SVD"] = DensityMatrices.load(dm_path + "/b2d-wiki10/bert2dm-svd.model")
    #
    # Let BERT2DM model substitute missing words
    b2d_models = ["BERT2DM PCA", "BERT2DM SVD"]
    missing_words = {"demilitarise": "demilitarised"}
    for model_name in b2d_models:
        for word in missing_words:
            dm_models[model_name].vocab.stoi[word] = dm_models[model_name].vocab.stoi[missing_words[word]]

    models = list(dm_models.values())
    model_names = list(dm_models.keys())
    dm_comps = ["verb_dm", "mult", "add", "kronecker", "phaser_verb"] #"phaser_noun", "fuzz_verb", "fuzz_noun"]
    dm_methods = [name + "_" + comp for name in model_names for comp in dm_comps]

    params = {"dm_comps": dm_comps, "dm_methods": dm_methods, "vector_embeddings": vector_embeddings,
              "vector_comps": vector_comps, "vector_methods": vector_methods,
              "pickle_embedding_files": pickle_embedding_files, "pickle_tensor_files": pickle_tensor_files,
              "sentences_path": args.sentences_path}

    # Evaluate on datasets
    eval_path = args.eval_path
    eval_datasets = ["ml2008", "gs2011", "gs2012", "ks2013-conll"]
    for dataset in eval_datasets:
        dataset_path = eval_path + "/" + EVAL_FILENAMES[dataset]
        results_df, ttest_df = evaluate(models, model_names, args, dataset, dataset_path, params)
        print_results(dataset, results_df, ttest_df, model_names, params)


def remove_zeros(num):
    if num.startswith("0."):
        return num[1:]
    elif num.startswith("-0."):
        return "-" + num[2:]
    return num


def print_results(name, results_df, ttest_df, dm_names, params):

    dm_comps = params["dm_comps"]
    vector_embeddings = params["vector_embeddings"]
    vector_comps = params["vector_comps"]
    spearman_col = "spearman"

    # Print vector results
    vector_results = {}
    max_score = -1.0
    max_embedding = None
    max_comp = None
    for embedding in vector_embeddings:
        vector_results[embedding] = []
        for comp in vector_comps:
            score = results_df.loc[embedding + "_" + comp][spearman_col]
            if score > max_score:
                max_score = score
                max_embedding = embedding
                max_comp = comp
            vector_results[embedding].append(score)
    vector_results_df = pd.DataFrame.from_dict(vector_results, orient="index", columns=vector_comps)

    # Print DM results
    dm_results = {}
    max_scores = {}
    max_comps = {}
    for dm in dm_names:
        dm_results[dm] = []
        max_scores[dm] = -1.0
        max_comps[dm] = None
        for comp in dm_comps:
            score = results_df.loc[dm + "_" + comp][spearman_col]
            if score > max_scores[dm]:
                max_scores[dm] = score
                max_comps[dm] = comp
            dm_results[dm].append(score)
    dm_results_df = pd.DataFrame.from_dict(dm_results, orient="index", columns=dm_comps)

    print("-----------------------------------------------------------")
    print(name)
    print("-----------------------------------------------------------")

    EMBEDDING_ABR = {"word2vec":"Word2Vec", "glove":"GloVe", "fasttext":"FastText"}
    for embedding, row in vector_results_df.iterrows():
        row_print = EMBEDDING_ABR[embedding]
        for comp in vector_results_df.columns:
            col_print = remove_zeros("{:.3f}".format(row[comp]))
            if comp == max_comp and embedding == max_embedding:
                col_print = "\\textbf{" + col_print + "}"
            row_print += " & " + col_print
        row_print += " \\\\"
        print(row_print)

    print("\\midrule")

    # Print BERT results
    if eval_bert:
        bert_score = results_df.loc["bert"]["spearman"]
        bert_score = remove_zeros("{:.3f}".format(bert_score))
        print("BERT & \\multicolumn{5}{c}{%s} \\\\" % bert_score)

    # Print InferSent results
    if eval_infersent:
        infersent_score = results_df.loc["infersent1"]["spearman"]
        infersent_score = remove_zeros("{:.3f}".format(infersent_score))
        print("InferSent1 & \\multicolumn{5}{c}{%s} \\\\" % infersent_score)

        infersent_score = results_df.loc["infersent2"]["spearman"]
        infersent_score = remove_zeros("{:.3f}".format(infersent_score))
        print("InferSent2 & \\multicolumn{5}{c}{%s} \\\\" % infersent_score)

    alpha = 0.05 / len(dm_results_df.index) # Bonferroni correction

    print("\\midrule")
    for dm, row in dm_results_df.iterrows():
        row_print = dm
        for comp in dm_results_df.columns:
            col_print = remove_zeros("{:.3f}".format(row[comp]))
            if col_print.startswith("."):
                col_print = "\\phantom{-}" + col_print
            if comp == max_comps[dm]:
                col_print = "\\textbf{" + col_print + "}"

                p_value = ttest_df.loc[dm + "_" + comp, max_embedding + "_" + max_comp]
                if max_scores[dm] > max_score and p_value <= alpha/2:
                    col_print = "\\underline{" + col_print + "}"
                elif p_value > alpha/2:
                    col_print += "\\underline{" + col_print + "}"

            row_print += " & " + col_print
        row_print += " \\\\"
        print(row_print)

    print("-----------------------------------------------------------")
    print("BEST TO WORST:")
    print(results_df.sort_values(by=["spearman"], ascending=False))
    print("-----------------------------------------------------------")


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dm_path", default="../../dms")
    parser.add_argument("--pretrained_path", default="../../pretrained")
    parser.add_argument("--eval_path", default="../evaluation")
    parser.add_argument("--sentences_path", default="../evaluation/complete_eval_sentences")
    args = parser.parse_args()
    main(args)