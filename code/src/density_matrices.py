import numpy as np
import math
import io
import pickle

class DensityMatrices(object):

    def __init__(self, vocab, dm):
        """
        vocab: torchtext FIELD.vocab object
        dm: numpy array containing density matrices, shape (vocab_size, dim, dim)
        """
        self.vocab = vocab
        self.vocab_size = len(vocab.itos)
        self.dm = dm
        self.dim = dm.shape[1]

    def normalise(self):
        # Normalise to unit trace
        self.dm = self.dm / np.reshape(self.dm.trace(axis1=1, axis2=2), (self.dm.shape[0], 1, 1))

    def contains(self, word):
        return word in self.vocab.stoi

    def get_dm(self, word):
        word_index = self.vocab.stoi[word]
        return self.dm[word_index]

    def similarity(self, word1, word2):
        word1_dm = self.get_dm(word1)
        word2_dm = self.get_dm(word2)
        # Efficient way to compute trace of matrix product
        trace = (word1_dm * word2_dm.T).sum().item()
        # Normalise
        trace = trace / (math.sqrt((word1_dm**2).sum()) * math.sqrt((word2_dm**2).sum()))
        return trace

    def sequence_similarity(self, sequence1, sequence2, methods, pos_tags):
        sims = {}
        for method in methods:
            dm1 = self.compose(sequence1, method, pos_tags)
            dm2 = self.compose(sequence2, method, pos_tags)
            if dm1 is None or dm2 is None:
                sims[method] = None
            else:
                trace = (dm1 * dm2.T).sum().item()
                normalised_inner_product = trace / (math.sqrt((dm1 ** 2).sum()) * math.sqrt((dm2 ** 2).sum()))
                sims[method] = normalised_inner_product
        return sims

    def compose(self, sequence, method, pos_tags):
        # sequence is a list of tokens to be composed in order
        if method == "verb_dm":
            result = self.get_dm(sequence[pos_tags.index("verb")])
        elif method == "mult":
            result = self.elementwise_mult(sequence)
        elif method == "add":
            result = self.add(sequence)
        elif method.startswith("fuzz"):
            result = self.fuzz_phaser(sequence, "fuzz", operator=method[method.find("_") + 1:], pos_tags=pos_tags)
        elif method.startswith("phaser"):
            result = self.fuzz_phaser(sequence, "phazer", operator=method[method.find("_") + 1:], pos_tags=pos_tags)
        elif method == "kronecker":
            result = self.kronecker_comp(sequence, pos_tags=pos_tags)

        # Normalise composed matrix
        result = result / result.trace()
        return result

    def save(self, file_path):
        with open(file_path, "wb") as model_file:
            pickle.dump(self, model_file, protocol=4)

    def kronecker_comp(self, sequence, pos_tags):
        if len(sequence) == 2:
            # Either subj verb or obj verb
            verb_dm = self.get_dm(sequence[pos_tags.index("verb")])
            noun_dm = self.get_dm(sequence[pos_tags.index("noun")])

            result = noun_dm * verb_dm

        elif len(sequence) == 3:
            # subj verb obj, to be composed ( subj (verb obj) )
            subj_dm = self.get_dm(sequence[0])
            verb_dm = self.get_dm(sequence[1])
            obj_dm = self.get_dm(sequence[2])

            # np.kron vs np.multiply.outer
            result = np.kron(subj_dm, obj_dm) * np.kron(verb_dm, verb_dm)

        elif len(sequence) == 5:
            # adj subj verb adj obj, to be composed ( (adj subj) (verb (adj obj) ) )
            adj_subj_dm = self.get_dm(sequence[0])
            subj_dm = self.get_dm(sequence[1])
            verb_dm = self.get_dm(sequence[2])
            adj_obj_dm = self.get_dm(sequence[3])
            obj_dm = self.get_dm(sequence[4])

            subj_np_dm = adj_subj_dm * subj_dm
            obj_np_dm = adj_obj_dm * obj_dm

            result = np.kron(subj_np_dm, obj_np_dm) * np.kron(verb_dm, verb_dm)

        return result

    """Composition methods"""
    # TODO: Normalise after each sub-composition
    def elementwise_mult(self, sequence):
        result = np.ones(shape=(self.dim, self.dim))
        for token in sequence:
            result *= self.get_dm(token)
        return result

    def add(self, sequence):
        result = np.zeros(shape=(self.dim, self.dim))
        for token in sequence:
            result += self.get_dm(token)
        return result

    def fuzz_phaser(self, sequence, composer, operator, pos_tags):

        if composer == "fuzz":
            compose_func = self.kmult
        else:
            compose_func = self.bmult

        if len(sequence) == 2:
            # Either subj verb or obj verb
            verb_dm = self.get_dm(sequence[pos_tags.index("verb")])
            noun_dm = self.get_dm(sequence[pos_tags.index("noun")])

            if operator == "verb":
                result = compose_func(operator_dm=verb_dm, input_dm=noun_dm)
            elif operator == "noun":
                result = compose_func(operator_dm=noun_dm, input_dm=verb_dm)

        elif len(sequence) == 3:
            # subj verb obj, to be composed ( subj (verb obj) )
            subj_dm = self.get_dm(sequence[0])
            verb_dm = self.get_dm(sequence[1])
            obj_dm = self.get_dm(sequence[2])

            if operator == "verb":
                vp_result = compose_func(operator_dm=verb_dm, input_dm=obj_dm)
                result = compose_func(operator_dm=vp_result, input_dm=subj_dm)
            elif operator == "noun":
                vp_result = compose_func(operator_dm=obj_dm, input_dm=verb_dm)
                result = compose_func(operator_dm=subj_dm, input_dm=vp_result)

        elif len(sequence) == 5:
            # adj subj verb adj obj, to be composed ( (adj subj) (verb (adj obj) ) )
            adj_subj_dm = self.get_dm(sequence[0])
            subj_dm = self.get_dm(sequence[1])
            verb_dm = self.get_dm(sequence[2])
            adj_obj_dm = self.get_dm(sequence[3])
            obj_dm = self.get_dm(sequence[4])

            subj_np_dm = compose_func(operator_dm=adj_subj_dm, input_dm=subj_dm)
            obj_np_dm = compose_func(operator_dm=adj_obj_dm, input_dm=obj_dm)

            if operator == "verb":
                vp_result = compose_func(operator_dm=verb_dm, input_dm=obj_np_dm)
                result = compose_func(operator_dm=vp_result, input_dm=subj_np_dm)
            elif operator == "noun":
                vp_result = compose_func(operator_dm=obj_np_dm, input_dm=verb_dm)
                result = compose_func(operator_dm=subj_np_dm, input_dm=vp_result)

        return result

    @staticmethod
    def load(file_path):
        return renamed_load(open(file_path, 'rb'))

    @staticmethod
    def bmult(operator_dm, input_dm):
        operator_dm_sqrt = DensityMatrices.get_dm_sqrt(operator_dm) # ugly OOP?
        result = operator_dm_sqrt @ input_dm @ operator_dm_sqrt
        result = result / result.trace()
        return result

    @staticmethod
    def get_dm_sqrt(dm):
        eigvals, eigvecs = np.linalg.eigh(dm)
        eigvals[eigvals <= 0] = 0
        dm_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T # element-wise multiplication in bracket achieves matrix multiplication with diagonal matrix
        return dm_sqrt

    @staticmethod
    def kmult(operator_dm, input_dm):
        eigvals, eigvecs = np.linalg.eigh(operator_dm)
        eigvals[eigvals <= 0] = 0
        result = np.zeros(shape=operator_dm.shape)
        for i in range(operator_dm.shape[0]):
            if eigvals[i] > 0:
                eigvec_outer = np.outer(eigvecs.T[i, :], eigvecs.T[i, :])
                result += eigvals[i] * (eigvec_outer @ input_dm @ eigvec_outer)
        result = result / result.trace()
        return result


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "word2dm":
            renamed_module = "density_matrices"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)


if __name__=="__main__":
    A = np.array([[3, 1], [1, 3]])
    B = np.array([[49, 2], [12, 94]])
    print(DensityMatrices.fuzz(A, B))
