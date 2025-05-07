from dataflow.core import TextRefiner
from dataflow.data import TextDataset
from dataflow.utils.registry import PROCESSOR_REGISTRY
from tqdm import tqdm
from dataflow.utils.utils import get_logger


@PROCESSOR_REGISTRY.register()
class RemoveExtraSpacesRefiner(TextRefiner):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.logger = get_logger()
        self.refiner_name = 'RemoveExtraSpacesRefiner'
        self.logger.info(f"Initializing {self.refiner_name}...")

    def refine_func(self, dataset):
        self.logger.info(f"Start running {self.refiner_name}...")
        refined_data = []
        numbers = 0
        keys = dataset.keys if isinstance(dataset.keys, list) else [dataset.keys]

        for item in tqdm(dataset, desc=f"Implementing {self.refiner_name}"):
            if isinstance(item, dict):
                modified = False
                for key in keys:
                    if key in item and isinstance(item[key], str):
                        original_text = item[key]
                        no_extra_spaces_text = " ".join(original_text.split())
                        
                        if original_text != no_extra_spaces_text:
                            item[key] = no_extra_spaces_text
                            modified = True
                            self.logger.debug(f"Modified text for key '{key}': Original: {original_text[:30]}... -> Refined: {no_extra_spaces_text[:30]}...")

                refined_data.append(item)
                if modified:
                    numbers += 1
                    self.logger.debug(f"Item modified, total modified so far: {numbers}")

        dataset.dataset = refined_data
        self.logger.info(f"Refining completed. Total items modified: {numbers}")
        return dataset, numbers
