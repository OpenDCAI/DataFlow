from dataflow.format import TextFormatter
from dataflow.utils.utils import get_logger

class Refiner():

    def __init__(self, args):
        pass

    def __call__(self, dataset):
        pass

class TextRefiner(Refiner):

    def __init__(self, args=None):
        self.data_type = "text"
        self.logger = get_logger()
        if "input_file" in args.keys():
            self.config = args
            self.formatter = TextFormatter(args)
            self.dataset = self.formatter.load_dataset()
        

        
    def __call__(self, dataset):
        refined_dataset, numbers = self.refine_func(dataset)
        self.logger.info(f'Implemented {self.refiner_name}. {numbers} data refined.')
        
        return refined_dataset
    
    def run(self):
        refined_dataset = self.__call__(self.dataset)
        refined_dataset.dump(self.config['output_file'])
