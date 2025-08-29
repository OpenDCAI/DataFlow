import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from googleapiclient import discovery
from dataflow.core import LLMServingABC

class PerspectiveAPIServing(LLMServingABC):
    """Service adapter for Google Perspective API."""
    def __init__(self, max_workers: int = 10):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if self.api_key is None:
            raise ValueError("Lack of Google API_KEY")
        self.max_workers = max_workers
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )


    def _call_api(self, text: str) -> float:
        """Invoke the Perspective API for a single text chunk and return toxicity score."""
        analyze_request = {
            'comment': { 'text': text },
            'requestedAttributes': {'TOXICITY': {}}
        }

        response = self.client.comments().analyze(body=analyze_request).execute()
        # extract the first span score
        return response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']

    def generate_from_input(self, user_inputs: list[str]) -> list[float]:  # type: ignore
        """
        Process a list of input texts concurrently and return toxicity scores.
        """
        scores: dict[int, float] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._call_api, text): idx
                       for idx, text in enumerate(user_inputs)}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring"): 
                idx = futures[fut]
                try:
                    scores[idx] = fut.result()
                except Exception as e:
                    scores[idx] = float('nan')
        return [scores[i] for i in range(len(user_inputs))]

    def cleanup(self) -> None:
        """Cleanup any resources or open connections if necessary."""
        # No persistent resources to clean up
        return