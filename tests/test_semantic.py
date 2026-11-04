import numpy as np

from aclkv.semantic import CharNgramEncoder, SemanticScorer


class StubEncoder:
    """Maps known strings to fixed unit vectors so thresholds are testable."""

    name = "stub"

    def __init__(self, table):
        self.table, self.calls = table, 0

    def encode(self, texts):
        self.calls += 1
        out = np.array([self.table[t] for t in texts], dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def test_identical_after_normalisation_is_one():
    s = SemanticScorer(CharNgramEncoder())
    assert s.similarity([("The Pentagon", "Pentagon")])[0] > 0.999


def test_yes_no_mismatch_is_zero_even_if_embeddings_are_close():
    enc = StubEncoder({"yes": [1, 0], "no": [0.99, 0.1]})
    s = SemanticScorer(enc)
    assert s.similarity([("no", "yes")]) == [0.0]
    assert s.similarity([("yes", "yes")])[0] > 0.999


def test_empty_prediction_scores_zero():
    s = SemanticScorer(CharNgramEncoder())
    assert s.similarity([("", "Paris")]) == [0.0]


def test_overlapping_answer_closer_than_unrelated():
    s = SemanticScorer(CharNgramEncoder())
    close, far = s.similarity([("Barack Obama", "Obama"), ("Eiffel Tower", "Obama")])
    assert close > far


def test_evaluate_counts_rescued_paraphrases_and_uses_cache():
    enc = StubEncoder({"auto": [1, 0], "car": [0.95, 0.31], "banana": [0, 1]})
    s = SemanticScorer(enc, threshold=0.8)
    res = s.evaluate([("auto", "car"), ("banana", "car"), ("car", "car")])
    assert res["n"] == 3
    assert abs(res["sem_match"] - 2 / 3) < 1e-9      # auto~car and car==car pass, banana fails
    assert abs(res["sem_rescued"] - 1 / 3) < 1e-9    # only "auto" is accepted without being an exact match
    calls_after_first = enc.calls
    s.evaluate([("auto", "car")])
    assert enc.calls == calls_after_first            # embeddings are cached
