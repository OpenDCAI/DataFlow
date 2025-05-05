from dataflow.format import TextFormatter

class Refiner():

    def __init__(self, args):
        pass

    def __call__(self, dataset):
        pass

class TextRefiner(Refiner):

    def __init__(self, args=None):
        self.data_type = "text"
        if "input_file" in args.keys():
            self.config = args
            self.formatter = TextFormatter(args)
            self.dataset = self.formatter.load_dataset()
        

        
    def __call__(self, dataset):
        refined_dataset, numbers = self.refine_func(dataset)
        print(f'Implemented {self.refiner_name}. {numbers} data refined.', flush=True)
        
        return refined_dataset
    
    def run(self):
        refined_dataset = self.__call__(self.dataset)
        refined_dataset.dump(self.config['output_file'])
