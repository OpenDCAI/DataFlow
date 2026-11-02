import pandas as pd
import pytest
from simhash import Simhash

from dataflow.operators.general_text.filter.simhash_deduplicate_filter import (
    SimHashDeduplicateFilter,
    get_similarity,
)
from dataflow.utils.storage import FileStorage

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize(
    "fingerprint_size, one_bit_similarity",
    [(8, 0.875), (16, 0.9375), (64, 0.984375), (128, 0.9921875)],
)
def test_similarity_uses_the_full_fingerprint(fingerprint_size, one_bit_similarity):
    zero = Simhash(0, f=fingerprint_size)
    one = Simhash(1, f=fingerprint_size)
    high_bit = 1 << (fingerprint_size - 1)
    high = Simhash(high_bit, f=fingerprint_size)
    high_plus_one = Simhash(high_bit | 1, f=fingerprint_size)
    opposite = Simhash((1 << fingerprint_size) - 1, f=fingerprint_size)

    assert get_similarity(zero, one) == pytest.approx(one_bit_similarity)
    assert get_similarity(one, zero) == pytest.approx(one_bit_similarity)
    assert get_similarity(high, high_plus_one) == pytest.approx(one_bit_similarity)
    assert get_similarity(zero, zero) == 1.0
    assert get_similarity(zero, opposite) == 0.0


@pytest.mark.parametrize("input_arg", [{"input_key": "text"}, {"input_keys": ["text"]}])
@pytest.mark.parametrize(
    "texts, bound, expected_texts",
    [
        (["world", "quality", "world"], 0.1, ["world", "quality"]),
        (["hello", "seven", "hello"], 0.125, ["hello"]),
    ],
)
def test_deduplication_respects_the_bit_distance(
    tmp_path, input_arg, texts, bound, expected_texts
):
    source = tmp_path / "input.jsonl"
    pd.DataFrame({"text": texts}).to_json(source, orient="records", lines=True)
    storage = FileStorage(str(source), cache_path=str(tmp_path / "cache"))
    storage.step()

    # Both pairs differ by one bit: 128/192 and 18/16, respectively.
    assert Simhash(texts[0], f=8).distance(Simhash(texts[1], f=8)) == 1
    result = SimHashDeduplicateFilter(fingerprint_size=8, bound=bound).run(
        storage, output_key="dedup_label", **input_arg
    )

    written = storage.step().read()
    assert result == ["dedup_label"]
    assert written["text"].tolist() == expected_texts
    assert written["dedup_label"].tolist() == [1] * len(expected_texts)
