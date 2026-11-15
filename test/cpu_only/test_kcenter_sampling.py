"""K-center selection preserves its budget without reselecting centers."""

import pandas as pd
import pytest
import torch

from dataflow.operators.core_text import KCenterGreedyFilter
from dataflow.operators.core_text.filter.kcentergreedy_filter import KCenterGreedy
from dataflow.utils.storage import DataFlowStorage, FileStorage

pytestmark = pytest.mark.cpu


class OfflineEmbeddings:
    def __init__(self, values=None):
        self.values = values
        self.calls = []

    def generate_embedding_from_input(self, texts):
        self.calls.append(list(texts))
        if self.values is not None:
            return self.values
        return [[float(text)] for text in texts]


class MemoryStorage(DataFlowStorage):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.result = None

    def get_keys_from_dataframe(self):
        return self.dataframe.columns.tolist()

    def read(self, output_type):
        assert output_type == "dataframe"
        return self.dataframe.copy()

    def write(self, dataframe):
        self.result = dataframe.copy()
        return "memory://selected"


@pytest.mark.parametrize("size,count", [(77, 5), (22, 15), (3, 5), (3, 0), (0, 5)])
def test_public_filter_preserves_integer_budget(tmp_path, size, count):
    source = tmp_path / "source.csv"
    rows = pd.DataFrame({"content": [str(i) for i in range(size)], "id": range(size)})
    rows.to_csv(source, index=False)
    original_source = source.read_bytes()
    storage = FileStorage(str(source), cache_path=str(tmp_path / "cache")).step()
    serving = OfflineEmbeddings()
    torch.manual_seed(0)

    result = KCenterGreedyFilter(count, serving).run(storage)

    selected = pd.read_json(storage._get_cache_file_path(1), lines=True)
    assert result == ["content"]
    assert len(selected) == min(size, count)
    if len(selected):
        assert selected["id"].is_unique
    if not size or not count:
        assert serving.calls == []
    assert source.read_bytes() == original_source


def test_public_filter_handles_zero_distances_and_duplicate_frame_indices():
    storage = MemoryStorage(
        pd.DataFrame({"content": ["a", "b", "c"], "id": [0, 1, 2]}, index=[7, 7, 3])
    )
    serving = OfflineEmbeddings([[0.0], [-1e-6], [-1e-6]])
    torch.manual_seed(0)

    KCenterGreedyFilter(2, serving).run(storage)

    assert len(storage.result) == 2
    assert storage.result["id"].is_unique
    assert storage.result["id"].tolist() == sorted(storage.result["id"].tolist())
    assert storage.dataframe["id"].tolist() == [0, 1, 2]


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_filter_rejects_invalid_counts(count):
    with pytest.raises(ValueError, match="num_samples"):
        KCenterGreedyFilter(count, OfflineEmbeddings())


@pytest.mark.parametrize("shape", [(4, 1), (4, 1, 1)])
def test_existing_centers_are_excluded_and_initialize_all_distances(shape):
    embeddings = torch.tensor([0.0, 1.0, 2.0, 10.0]).reshape(shape)
    sampler = KCenterGreedy(embeddings, 0.25)
    already_selected = [0, 3]
    torch.manual_seed(1)

    assert sampler.select_coreset_idxs(already_selected) == [2]
    assert already_selected == [0, 3]


@pytest.mark.parametrize("shape", [(4, 1), (4, 1, 1)])
def test_new_selection_is_capped_to_remaining_capacity(shape):
    embeddings = torch.arange(4.0).reshape(shape)
    sampler = KCenterGreedy(embeddings, 1.0)
    torch.manual_seed(1)

    assert sorted(sampler.select_coreset_idxs([0, 0, 3])) == [1, 2]
    assert sampler.select_coreset_idxs([0, 1, 2, 3]) == []


@pytest.mark.parametrize(
    "embeddings,ratio",
    [(torch.empty(0, 2), 0.5), (torch.ones(3, 2), 0.0)],
)
def test_empty_selection_does_not_draw_a_random_anchor(monkeypatch, embeddings, ratio):
    def unexpected_draw(**kwargs):
        pytest.fail("An empty selection must not draw a random anchor")

    monkeypatch.setattr(torch, "randint", unexpected_draw)
    assert KCenterGreedy(embeddings, ratio).select_coreset_idxs() == []


@pytest.mark.parametrize("shape", [(8, 2), (8, 1, 2)])
def test_identical_embeddings_select_every_row_once_and_can_be_reused(shape):
    sampler = KCenterGreedy(torch.zeros(shape), 1.0)
    for _ in range(2):
        indices = sampler.select_coreset_idxs()
        assert sorted(indices) == list(range(8))


def test_unseeded_selection_keeps_the_random_anchor_strategy(monkeypatch):
    monkeypatch.setattr(torch, "randint", lambda **kwargs: torch.tensor([1]))
    sampler = KCenterGreedy(torch.tensor([[0.0], [1.0], [10.0]]), 0.5)
    assert sampler.select_coreset_idxs() == [2]


def test_zero_distance_ties_still_select_unique_indices():
    torch.manual_seed(0)
    sampler = KCenterGreedy(torch.tensor([[0.0], [-1e-6], [-1e-6]]), 1.0)
    assert sorted(sampler.select_coreset_idxs()) == [0, 1, 2]


@pytest.mark.parametrize("count", [1, 5])
def test_selection_does_not_compute_unused_final_distances(monkeypatch, count):
    from dataflow.operators.core_text.filter import kcentergreedy_filter as module

    pairwise_distance = module.F.pairwise_distance
    calls = []

    def counted_distances(*args, **kwargs):
        calls.append(None)
        return pairwise_distance(*args, **kwargs)

    monkeypatch.setattr(module.F, "pairwise_distance", counted_distances)
    sampler = KCenterGreedy(torch.arange(8.0).reshape(-1, 1), count / 8)
    assert len(sampler.select_coreset_idxs()) == count
    assert len(calls) == count
