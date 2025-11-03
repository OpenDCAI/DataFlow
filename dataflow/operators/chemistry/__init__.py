from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from generate.extract_smiles_from_text_generator import ExtractSmilesFromTextGenerator
    from eval.smiles_equivalence_dataset_evaluator import SmilesEquivalenceDatasetEvaluator
else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/chemistry/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/chemistry/", _import_structure)
