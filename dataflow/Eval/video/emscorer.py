import torch
import clip
import time
import numpy as np
from dataflow.core import VideoTextScorer
from dataflow.utils.registry import MODEL_REGISTRY
from collections import defaultdict
from .emscore.utils import em_cos_score, get_idf_dict

@MODEL_REGISTRY.register()
class EMScorer(VideoTextScorer):

    def __init__(self, args_dict: dict, cfg=None):
        super().__init__(args_dict)
        self.scorer_name = "EMScorer"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        self._model = model
        self._tokenizer = clip.tokenize
        self._image_preprocess = preprocess

    def init_score(self, len_dataset):
        return {'EMScore(X,X*)': {
                    'figr_P': np.array([np.nan] * len_dataset),
                    'figr_R': np.array([np.nan] * len_dataset),
                    'figr_F': np.array([np.nan] * len_dataset),
                    'cogr': np.array([np.nan] * len_dataset),
                    'full_P': np.array([np.nan] * len_dataset),
                    'full_R': np.array([np.nan] * len_dataset),
                    'full_F': np.array([np.nan] * len_dataset),
                    },
                'EMScore(X,V)': {
                    'figr_P': np.array([np.nan] * len_dataset),
                    'figr_R': np.array([np.nan] * len_dataset),
                    'figr_F': np.array([np.nan] * len_dataset),
                    'cogr': np.array([np.nan] * len_dataset),
                    'full_P': np.array([np.nan] * len_dataset),
                    'full_R': np.array([np.nan] * len_dataset),
                    'full_F': np.array([np.nan] * len_dataset),                    
                    },
                'EMScore(X,V,X*)':{
                    'figr_P': np.array([np.nan] * len_dataset),
                    'figr_R': np.array([np.nan] * len_dataset),
                    'figr_F': np.array([np.nan] * len_dataset),
                    'cogr': np.array([np.nan] * len_dataset),
                    'full_P': np.array([np.nan] * len_dataset),
                    'full_R': np.array([np.nan] * len_dataset),
                    'full_F': np.array([np.nan] * len_dataset),                    
                    },
                }

    def evaluate_batch(self, sample):
        # print(sample)
        cands = list(sample['captions'][0])
        refs = list(sample['captions'][1])
        vids = sample['video']
        # print(cands, refs, vids)

        ref_group_boundaries = None
        ori_cands, ori_refs = cands, refs
        # if reference are avaliable, and there are multiple references for each candidata caption
        if refs and not isinstance(refs[0], str):
            ref_group_boundaries = []
            cands, refs = [], []
            count = 0
            for cand, ref_group in zip(ori_cands, ori_refs):
                cands += [cand] * len(ref_group)
                refs += ref_group
                ref_group_boundaries.append((count, count + len(ref_group)))
                count += len(ref_group)

        idf_dict = defaultdict(lambda: 1.0)        
        print("calculating EMScore scores...")
        time_start = time.perf_counter()

        results = em_cos_score(
            self._model,
            refs,
            cands,
            ori_cands,
            ori_refs,
            vids,
            None,
            self._tokenizer,
            idf_dict,
            self._image_preprocess,
            verbose=True,
            device='cuda',
            batch_size=64,
            return_matched_idx=False,
        )
        
        if results is None:
            return None
        
        final_results = {}
        if refs:
            refs_all_local_preds  = results['refs_result']['figr']
            refs_all_global_preds = results['refs_result']['cogr']
            if ref_group_boundaries is not None:
                max_preds_local = []
                for start, end in ref_group_boundaries:
                    max_preds_local.append(refs_all_local_preds[start:end].max(dim=0)[0])
                refs_all_local_preds = torch.stack(max_preds_local, dim=0)

                max_preds_global = []
                for start, end in ref_group_boundaries:
                    max_preds_global.append(refs_all_global_preds[start:end].max())
                refs_all_global_preds = torch.stack(max_preds_global, dim=0)

            refs_P, refs_R, refs_F = refs_all_local_preds[..., 0], refs_all_local_preds[..., 1], refs_all_local_preds[..., 2]  # P, R, F
            
            refs_results = {}
            refs_results['figr_P'] = refs_P
            refs_results['figr_R'] = refs_R
            refs_results['figr_F'] = refs_F
            refs_results['cogr'] = refs_all_global_preds
            refs_results['full_P'] = (refs_results['figr_P'] + refs_results['cogr'])/2
            refs_results['full_R'] = (refs_results['figr_R'] + refs_results['cogr'])/2
            refs_results['full_F'] = (refs_results['figr_F'] + refs_results['cogr'])/2
            # refs_results['refs_matched_indices'] = results['refs_result']['matched_indices']
            final_results['EMScore(X,X*)'] = refs_results
        
        if vids:
            vid_all_local_preds  = results['vid_result']['figr']
            vid_all_global_preds = results['vid_result']['cogr']
            vid_P, vid_R, vid_F  = vid_all_local_preds[..., 0], vid_all_local_preds[..., 1], vid_all_local_preds[..., 2]   # P, R, F

            vid_results = {}
            vid_results['figr_P'] = vid_P
            vid_results['figr_R'] = vid_R
            vid_results['figr_F'] = vid_F
            vid_results['cogr'] = vid_all_global_preds
            vid_results['full_P'] = (vid_results['figr_P'] + vid_results['cogr'])/2
            vid_results['full_R'] = (vid_results['figr_R'] + vid_results['cogr'])/2
            vid_results['full_F'] = (vid_results['figr_F'] + vid_results['cogr'])/2
            # vid_results['vid_matched_indices'] = results['vid_result']['matched_indices']
            final_results['EMScore(X,V)'] = vid_results
        
        if refs and vids:
            vid_refs_result = {}
            vid_refs_result['figr_P'] = (final_results['EMScore(X,V)']['figr_P'] + final_results['EMScore(X,X*)']['figr_P'])/2
            vid_refs_result['figr_R'] = (final_results['EMScore(X,V)']['figr_R'] + final_results['EMScore(X,X*)']['figr_R'])/2
            vid_refs_result['figr_F'] = (final_results['EMScore(X,V)']['figr_F'] + final_results['EMScore(X,X*)']['figr_F'])/2
            vid_refs_result['cogr'] = (final_results['EMScore(X,V)']['cogr'] + final_results['EMScore(X,X*)']['cogr'])/2
            vid_refs_result['full_P'] = (vid_refs_result['figr_P'] + vid_refs_result['cogr'])/2
            vid_refs_result['full_R'] = (vid_refs_result['figr_R'] + vid_refs_result['cogr'])/2
            vid_refs_result['full_F'] = (vid_refs_result['figr_F'] + vid_refs_result['cogr'])/2
            final_results['EMScore(X,V,X*)'] = vid_refs_result


        time_diff = time.perf_counter() - time_start
        print(f"done in {time_diff:.2f} seconds, {len(cands) / time_diff:.2f} sentences/sec")

        return final_results
