"""Image quality and duplicate filtering with configurable thresholds."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class EstimatedImagePolicy:
    min_short_side: int = 32
    max_aspect_ratio: float = 20.0
    min_gray_std: float = 1.0
    min_laplacian_variance: float = 10.0
    phash_max_distance: int = 4
    clip_min_similarity: float = 0.995
    question_min_similarity: float = 0.98
    metric_long_side: int = 512
    provenance: str = 'default_image_thresholds_v1'

    def __post_init__(self):
        values = [self.min_short_side, self.max_aspect_ratio, self.min_gray_std,
                  self.min_laplacian_variance, self.clip_min_similarity,
                  self.question_min_similarity, self.metric_long_side]
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Policy thresholds must be finite')
        if (self.min_short_side < 1 or self.max_aspect_ratio < 1 or self.metric_long_side < 3
                or self.min_gray_std < 0 or self.min_laplacian_variance < 0
                or not 0 <= self.phash_max_distance <= 7
                or not 0 <= self.question_min_similarity <= 1
                or not -1 <= self.clip_min_similarity <= 1):
            raise ValueError('Invalid estimated image-filter policy')

    def manifest(self):
        return dict(values=asdict(self), status='heuristic_thresholds')


def image_metrics(path, metric_long_side=512):
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert('L')
        width, height = image.size
        image.thumbnail((metric_long_side, metric_long_side), Image.Resampling.BICUBIC)
        gray = np.asarray(image, dtype=np.float64)
        lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2]
               + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
        tiny = np.asarray(image.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    x = np.arange(32)
    cosine = np.cos(np.pi * (2 * x[None, :] + 1) * x[:, None] / 64)
    low = (cosine @ tiny @ cosine.T)[:8, :8].ravel()
    bits = low > np.median(low[1:])
    phash = sum(int(bit) << i for i, bit in enumerate(bits))
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return dict(width=width, height=height, short_side=min(width, height),
                aspect_ratio=max(width, height) / min(width, height),
                gray_std=float(gray.std()),
                laplacian_variance=float(lap.var()) if lap.size else 0.0,
                phash=phash, image_sha256=h.hexdigest())


def projected_clip_output(output):
    # Current Transformers returns BaseModelOutputWithPooling; older releases
    # returned a Tensor directly. Never use token hidden states as image vectors.
    features = getattr(output, 'pooler_output', output)
    if getattr(features, 'ndim', None) != 2:
        raise RuntimeError('Expected batched projected CLIP image features')
    return features


class BestEffortImageFilter:
    """SHA-256 + pHash + CLIP near-duplicate and quality filtering.

    Keeps different questions about the same image. CLIP backbone, thresholds,
    question similarity, and retrieval are documented estimates. For >4096 rows
    FAISS is required; no silent quadratic scan of a large dataset is allowed.
    """
    def __init__(self, encoder, policy=None, use_faiss=True):
        if encoder is None:
            raise ValueError('A CLIP image encoder is required; no silent CLIP-stage omission')
        self.encoder = encoder
        self.policy = policy or EstimatedImagePolicy()
        self.use_faiss = use_faiss
        self.reset()

    def reset(self):
        self.exact = set()
        self.records = []
        self.sha_to_indices = {}
        self.phash_buckets = [dict() for _ in range(8)]
        self.index = None

    def manifest(self):
        return dict(policy=self.policy.manifest(), clip_model=getattr(self.encoder, 'model_id', 'unspecified'),
                    mocked=bool(getattr(self.encoder, 'mocked', False)),
                    near_duplicate_method='similar_question AND (same_SHA256 OR near_pHash OR near_CLIP)',
                    retrieval='FAISS HNSW top64 when installed; exact small-pool cosine otherwise',
                    similarity='CLIP cosine and text similarity')

    def evaluate(self, path, question, answer):
        try:
            metric = image_metrics(path, self.policy.metric_long_side)
        except (OSError, ValueError) as exc:
            return dict(keep=False, reason='unreadable_image', error=type(exc).__name__)
        p = self.policy
        for condition, reason in [
            (metric['short_side'] < p.min_short_side, 'estimated_low_resolution'),
            (metric['aspect_ratio'] > p.max_aspect_ratio, 'estimated_extreme_aspect_ratio'),
            (metric['gray_std'] < p.min_gray_std, 'estimated_blank_image'),
            (metric['laplacian_variance'] < p.min_laplacian_variance, 'estimated_blurry_image')]:
            if condition:
                return dict(keep=False, reason=reason, metrics=metric)
        exact = (metric['image_sha256'], question, answer)
        if exact in self.exact:
            return dict(keep=False, reason='exact_image_QA_duplicate', metrics=metric)
        vector = np.asarray(self.encoder.encode(path), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or norm <= 0:
            raise RuntimeError('Invalid CLIP embedding')
        vector = vector / norm
        if self.index is None and self.use_faiss:
            try:
                import faiss
                self.index = faiss.IndexHNSWFlat(len(vector), 32, faiss.METRIC_INNER_PRODUCT)
                self.index.hnsw.efConstruction = 100
                self.index.hnsw.efSearch = 128
            except ImportError:
                self.use_faiss = False
        if not self.use_faiss and len(self.records) >= 4096:
            raise RuntimeError('Install faiss-cpu for large-pool CLIP retrieval; small exact index limit reached')
        candidates = set(self.sha_to_indices.get(metric['image_sha256'], []))
        # 8 byte bands find all pHash pairs with <=4 differing bits in some unchanged band.
        for band, buckets in enumerate(self.phash_buckets):
            candidates.update(buckets.get((metric['phash'] >> (band * 8)) & 255, []))
        if self.index is not None and self.records:
            _, indices = self.index.search(vector[None], min(64, len(self.records)))
            candidates.update(int(i) for i in indices[0] if i >= 0)
        elif self.records:
            candidates.update(range(len(self.records)))
        for i in sorted(candidates):
            previous = self.records[i]
            similarity = SequenceMatcher(None, question, previous['question'], autojunk=False).ratio()
            if similarity < p.question_min_similarity:
                continue
            hamming = (metric['phash'] ^ previous['phash']).bit_count()
            cosine = float(vector @ previous['vector'])
            if (metric['image_sha256'] == previous['sha'] or hamming <= p.phash_max_distance
                    or cosine >= p.clip_min_similarity):
                return dict(keep=False, reason='estimated_near_duplicate', metrics=metric,
                            prior_index=i, question_similarity=similarity,
                            phash_distance=hamming, clip_similarity=cosine)
        index = len(self.records)
        self.exact.add(exact)
        self.records.append(dict(question=question, vector=vector, phash=metric['phash'], sha=metric['image_sha256']))
        self.sha_to_indices.setdefault(metric['image_sha256'], []).append(index)
        for band, buckets in enumerate(self.phash_buckets):
            buckets.setdefault((metric['phash'] >> (band * 8)) & 255, []).append(index)
        if self.index is not None:
            self.index.add(vector[None])
        return dict(keep=True, reason='raw_filter_passed', metrics=metric)
