import numpy
from collections import deque
import string
numpy.random.seed(12345)


class InputData:
    """Store data for word2vec, such as word map, sampling table and so on.

    Attributes:
        word_frequency: Count of each word, used for filtering low-frequency words and sampling table
        word2id: Map from word to word id, without low-frequency words.
        id2word: Map from word id to word, without low-frequency words.
        sentence_count: Sentence count in files.
        word_count: Word count in files, without low-frequency words.
    """

    def __init__(self, file_name, min_count):
        self.input_file_name = file_name
        self.get_words(min_count)
        self.subsample_t = 1e-5
        self.word_pair_catch = deque()
        self.cbow_pair_catch = deque()
        self.init_sample_table()
        self.init_discard_table()
        print('Word Count: %d' % len(self.word2id))
        print('Sentence Length: %d' % (self.sentence_length))

    def _sentence_to_word_ids(self, sentence):
        if not sentence:
            return []
        words = sentence.strip().split()
        if not words:
            return []
        return [
            self.word2id[word]
            for word in (self._normalize_token(token) for token in words)
            if word in self.word2id
        ]

    def _normalize_token(self, token):
        if not token:
            return ""
        return token.strip(string.punctuation + "“”‘’")

    def get_words(self, min_count):
        self.input_file = open(self.input_file_name, encoding='utf-8')
        self.sentence_length = 0
        self.sentence_count = 0
        word_frequency = dict()
        for line in self.input_file:
            self.sentence_count += 1
            words = [self._normalize_token(token) for token in line.strip().split()]
            words = [word for word in words if word]
            if not words:
                continue
            self.sentence_length += len(words)
            for w in words:
                word_frequency[w] = word_frequency.get(w, 0) + 1
        self.word2id = dict()
        self.id2word = dict()
        wid = 0
        self.word_frequency = dict()
        for w, c in word_frequency.items():
            if c < min_count:
                self.sentence_length -= c
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        self.word_count = len(self.word2id)
        self.input_file = open(self.input_file_name, encoding='utf-8')

    def init_sample_table(self):
        pow_frequency = numpy.array(list(self.word_frequency.values()), dtype=numpy.float64) ** 0.75
        words_pow = pow_frequency.sum()
        if words_pow == 0:
            self.sample_ids = numpy.array([], dtype=numpy.int64)
            self.sample_prob = numpy.array([], dtype=numpy.float64)
            return
        self.sample_ids = numpy.arange(self.word_count, dtype=numpy.int64)
        self.sample_prob = pow_frequency / words_pow

    def init_discard_table(self):
        if self.sentence_length <= 0:
            self.discard_prob = numpy.array([], dtype=numpy.float64)
            return
        freq = numpy.array(
            [self.word_frequency[wid] for wid in range(self.word_count)],
            dtype=numpy.float64,
        )
        freq = freq / float(self.sentence_length)
        threshold = float(self.subsample_t)
        # Standard word2vec subsampling: keep_prob = (sqrt(f / t) + 1) * (t / f)
        keep_prob = (numpy.sqrt(freq / threshold) + 1.0) * (threshold / freq)
        keep_prob = numpy.where(freq > 0.0, keep_prob, 0.0)
        self.discard_prob = numpy.clip(1.0 - keep_prob, 0.0, 1.0)

    def _keep_token(self, word_id):
        if not hasattr(self, 'discard_prob') or self.discard_prob.size == 0:
            return True
        return numpy.random.random() >= self.discard_prob[word_id]

    # @profile
    def get_batch_pairs(self, batch_size, window_size):
        while len(self.word_pair_catch) < batch_size:
            sentence = self.input_file.readline()
            if sentence is None or sentence == '':
                self.input_file = open(self.input_file_name, encoding='utf-8')
                sentence = self.input_file.readline()
            word_ids = [word_id for word_id in self._sentence_to_word_ids(sentence) if self._keep_token(word_id)]
            for i, u in enumerate(word_ids):
                for j, v in enumerate(
                        word_ids[max(i - window_size, 0):i + window_size + 1]):
                    assert u < self.word_count
                    assert v < self.word_count
                    if i == j:
                        continue
                    self.word_pair_catch.append((u, v))
        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())
        return batch_pairs

    def get_cbow_batch(self, batch_size, window_size):
        while len(self.cbow_pair_catch) < batch_size:
            sentence = self.input_file.readline()
            if sentence is None or sentence == '':
                self.input_file = open(self.input_file_name, encoding='utf-8')
                sentence = self.input_file.readline()
            word_ids = [word_id for word_id in self._sentence_to_word_ids(sentence) if self._keep_token(word_id)]
            for i, target in enumerate(word_ids):
                context = word_ids[max(i - window_size, 0):i] + word_ids[i + 1:i + 1 + window_size]
                if len(context) == 0:
                    continue
                self.cbow_pair_catch.append((context, target))

        batch_contexts = []
        batch_targets = []
        max_len = 0
        for _ in range(batch_size):
            context, target = self.cbow_pair_catch.popleft()
            batch_contexts.append(context)
            batch_targets.append(target)
            if len(context) > max_len:
                max_len = len(context)

        pad_id = self.word_count
        padded_contexts = [context + [pad_id] * (max_len - len(context)) for context in batch_contexts]
        return padded_contexts, batch_targets

    # @profile
    def get_neg_v_neg_sampling(self, pos_word_pair, count):
        neg_v = numpy.random.choice(
            self.sample_ids, size=(len(pos_word_pair), count), p=self.sample_prob).tolist()
        return neg_v

    def evaluate_pair_count(self, window_size):
        return self.sentence_length * (2 * window_size - 1) - (
            self.sentence_count - 1) * (1 + window_size) * window_size

    def evaluate_cbow_count(self):
        return self.sentence_length


def test():
    a = InputData('./zhihu.txt')


if __name__ == '__main__':
    test()
