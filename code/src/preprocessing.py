import xml.etree.ElementTree as ET
import numpy as np
import os
import argparse
from density_matrices import DensityMatrices


def vu_amsterdam(input_path, output_path):
    tree = ET.parse(input_path)
    root = tree.getroot()
    tei = "{http://www.tei-c.org/ns/1.0}"
    count = 0

    output_file = open(output_path, "w")
    output_file.close()
    output_file = open(output_path, "a")

    for s in root.iter(tei + "s"):
        sentence = ""
        for token in s:
            count+=1
            if token.text is not None:
                sentence += token.text
        sentence = " ".join(sentence.split())
        output_file.write(sentence + "\n")
    #print(count)
    output_file.close()


def wackypedia(input_path, output_path):
    output_file = open(output_path, "w")
    output_file.close()
    output_file = open(output_path, "a")

    with open(input_path, encoding="latin-1") as file:
        sentence_tokens = []
        for line in file:
            if line.startswith("<text"):
                continue
            elif line.strip() == "<s>":
                sentence_tokens = []
            elif line.strip() == "</s>":
                sentence = " ".join(sentence_tokens)
                output_file.write(sentence + "\n")
            else:
                word = line.split()[0]
                sentence_tokens.append(word)


def ukwac(input_path, output_path):

    for file_name in os.listdir(input_path):
        if file_name.startswith("UKWAC"):
            input_file_path = os.path.join(input_path, file_name)

            output_file_path = output_path + "/" + file_name[0:file_name.index(".")] + "_preproc"
            output_file = open(output_file_path, "w")
            output_file.close()
            output_file = open(output_file_path, "a")

            with open(input_file_path, encoding="latin-1") as file:
                sentence_tokens = []
                for line in file:
                    if line.startswith("<text"):
                        continue
                    elif line.strip() == "<s>":
                        sentence_tokens = []
                    elif line.strip() == "</s>":
                        sentence = " ".join(sentence_tokens)
                        output_file.write(sentence + "\n")
                    else:
                        word = line.split()[0]
                        sentence_tokens.append(word)

            print("Preprocessed %s" % file_name)
            output_file.close()


def ukwacky(input_path, output_path):
    # Preprocess multiple input files and write to one output file.
    output_file = open(output_path, "w")
    output_file.close()
    output_file = open(output_path, "a")

    for file_name in os.listdir(input_path):
        if file_name.startswith("UKWAC") or file_name.startswith("wackypedia"):
            lemma_index = 1 if file_name.startswith("wackypedia") else 2
            input_file_path = os.path.join(input_path, file_name)
            with open(input_file_path, encoding="latin-1") as file:
                sentence_tokens = []
                for line in file:
                    if line.startswith("<text") or line.startswith("</text>"):
                        continue
                    elif line.strip() == "<s>":
                        sentence_tokens = []
                    elif line.strip() == "</s>":
                        sentence = " ".join(sentence_tokens)
                        output_file.write(sentence + "\n")
                    else:
                        words = line.split() # lemma instead of word
                        word = words[0]
                        sentence_tokens.append(word)
                        # if len(words) > lemma_index:
                        #     word = words[lemma_index]
                        #     sentence_tokens.append(word)
            print("Preprocessed %s" % file_name)

    output_file.close()


def clean_corpus(input_path, output_path):
    output_file = open(output_path, "w")
    output_file.close()
    output_file = open(output_path, "a")

    with open(input_path, encoding="latin-1") as file:
        for line in file:
            tokens = line.split()
            tokens = [token for token in tokens if token.isalpha()] # changed to token.isalpha() if you want
            if len(tokens) >= 2:
                sentence = " ".join(tokens) + "\n"
                output_file.write(sentence.lower()) # remove lower for corpi


def pregroup_clean_corpus(input_path, output_path):
    output_file = open(output_path, "w")
    output_file.close()
    output_file = open(output_path, "a")

    with open(input_path, encoding="latin-1") as file:
        for line in file:
            tokens = line.split()
            tokens = [token for token in tokens if token.isalnum()] # remove punctuation
            sent_len = len(tokens)
            tokens = [token for token in tokens if token.isalpha()]  # remove tokens with numbers

            if len(tokens) == sent_len and len(tokens) >= 3:
                sentence = " ".join(tokens) + "\n"
                output_file.write(sentence)


def remove_numbers(input_path, output_path):
    output_file = open(output_path, "w")
    output_file.close()
    output_file = open(output_path, "a")

    with open(input_path, encoding="latin-1") as file:
        for line in file:
            tokens = line.split()
            tokens = [token for token in tokens if token.isalpha()]
            if len(tokens) >= 2:
                sentence = " ".join(tokens) + "\n"
                output_file.write(sentence)


def save_dms_as_embeddings(dms, output_path):
    output_file = open(output_path, "w")
    output_file.close()
    output_file = open(output_path, "a")

    for word in dms.vocab.itos:
        dm = dms.get_dm(word)
        if not np.isnan(dm).any():
            dm_array = dm.flatten()
            embedding = ' '.join(str(num) for num in dm_array)
            output_file.write(word + " " + embedding + "\n")
    output_file.close()
    print("DMs saved successfully as embeddings.")


def main(input_path, output_path):
    clean_corpus(input_path, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="../../data/corpora/wackypedia10_preproc")
    parser.add_argument("--output_path", default="../../data/corpora/wackypedia10_clean")
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    main(input_path, output_path)