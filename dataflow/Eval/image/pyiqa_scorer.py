import pyiqa
from dataflow.core.scorer import ImageScorer
from dataflow.utils.registry import MODEL_REGISTRY
from ...utils.image_utils import pyiqa_image_preprocess

TYPE_KEY = "type"

# @MODEL_REGISTRY.register()
class PyiqaScorer(ImageScorer):
    def __init__(self, args_dict: dict, metric_name):
        super().__init__(args_dict)
        assert metric_name in pyiqa.list_models(metric_mode="NR"), f"Metric {metric_name} not available in PyIQA"
        self.model = pyiqa.create_metric(metric_name, device=args_dict["device"])
        self.image_preprocessor = pyiqa_image_preprocess
        self.data_type = "image"
        
    def evaluate_batch(self, sample):
        return self.model(sample)

@MODEL_REGISTRY.register()
class QalignScorer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict, 'qalign')
        self.scorer_name = "QalignScorer"

@MODEL_REGISTRY.register()
class LiqeScorer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        metric_name = 'liqe'
        if TYPE_KEY in args_dict:
            metric_name = metric_name + "_" + args_dict[TYPE_KEY]
        super().__init__(args_dict, metric_name)
        self.scorer_name = f"LiqeScorer({metric_name})"

@MODEL_REGISTRY.register()
class ArniqaScorer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        metric_name = 'arniqa'
        if TYPE_KEY in args_dict:
            metric_name = metric_name + "-" + args_dict[TYPE_KEY]
        super().__init__(args_dict, metric_name)
        self.scorer_name = f"ArniqaScorer({metric_name})"

@MODEL_REGISTRY.register()
class TopiqScorer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        metric_name = 'topiq'
        if TYPE_KEY in args_dict:
            metric_name = metric_name + "_" + args_dict[TYPE_KEY]
        else:
            raise ValueError("TOPIQ requires a type key in the YAML file")
        super().__init__(args_dict, metric_name)
        self.scorer_name = f"TopiqScorer({metric_name})"

@MODEL_REGISTRY.register()
class ClipiqaScorer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        metric_name = 'clipiqa'
        if TYPE_KEY in args_dict:
            metric_name = metric_name + args_dict[TYPE_KEY]
        super().__init__(args_dict, metric_name)
        self.scorer_name = f"ClipiqaScorer({metric_name})"

@MODEL_REGISTRY.register()
class ManiqaScorer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        metric_name = 'maniqa'
        if TYPE_KEY in args_dict:
            metric_name = metric_name + "-" + args_dict[TYPE_KEY]
        super().__init__(args_dict, metric_name)
        self.scorer_name = f"ManiqaScorer({metric_name})"

@MODEL_REGISTRY.register()
class MusiqScorer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        metric_name = 'musiq'
        if TYPE_KEY in args_dict:
            metric_name = metric_name + "-" + args_dict[TYPE_KEY]
        super().__init__(args_dict, metric_name)
        self.scorer_name = f"MusiqScorer({metric_name})"

@MODEL_REGISTRY.register()
class DbcnnScorer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        metric_name = 'dbcnn'
        super().__init__(args_dict, metric_name)
        self.scorer_name = "DbcnnScorer"

@MODEL_REGISTRY.register()
class Pqa2piqScorer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        metric_name = 'pqa2piqs'
        super().__init__(args_dict, metric_name)
        self.scorer_name = "Pqa2piqScorer"

@MODEL_REGISTRY.register()
class HyperiqaScorer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        metric_name = 'hyperiqa'
        super().__init__(args_dict, metric_name)
        self.scorer_name = "HyperiqaScorer"

@MODEL_REGISTRY.register()
class NimaScorer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        metric_name = 'nima'
        if TYPE_KEY in args_dict:
            metric_name = metric_name + "-" + args_dict[TYPE_KEY]
        super().__init__(args_dict, metric_name)
        self.scorer_name = f"NimaScorer({metric_name})"

@MODEL_REGISTRY.register()
class WadiqamScoreer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        metric_name = 'wadiqam_nr'
        super().__init__(args_dict, metric_name)
        self.scorer_name = "WadiqamScoreer"

@MODEL_REGISTRY.register()
class CnniqaScorer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        metric_name = 'cnniqa'
        super().__init__(args_dict, metric_name)
        self.scorer_name = "CnniqaScorer"

@MODEL_REGISTRY.register()
class NrqmScoreer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        metric_name = 'nrqm'
        super().__init__(args_dict, metric_name)
        self.scorer_name = "NrqmScoreer"

@MODEL_REGISTRY.register()
class PiScorer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        metric_name = 'pi'
        super().__init__(args_dict, metric_name)
        self.scorer_name = "PiScorer"

@MODEL_REGISTRY.register()
class BrisqueScorer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        metric_name = 'brisque'
        super().__init__(args_dict, metric_name)
        self.scorer_name = "BrisqueScorer"

@MODEL_REGISTRY.register()
class IlniqeScorer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        metric_name = 'ilniqe'
        super().__init__(args_dict, metric_name)
        self.scorer_name = "IlniqeScorer"

@MODEL_REGISTRY.register()
class NiqeScorer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        metric_name = 'niqe'
        super().__init__(args_dict, metric_name)
        self.scorer_name = "NiqeScorer"

@MODEL_REGISTRY.register()
class PiqeScorer(PyiqaScorer):
    def __init__(self, args_dict: dict):
        metric_name = 'piqe'
        super().__init__(args_dict, metric_name)
        self.scorer_name = "PiqeScorer"
