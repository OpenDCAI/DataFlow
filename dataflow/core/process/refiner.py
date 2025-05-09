class Refiner():

    def __init__(self, args):
        pass

    def __call__(self, dataset):
        pass

class TextRefiner(Refiner):

    def __init__(self, args=None):
        self.data_type = "text"
        
    def __call__(self, dataset):
        refined_dataset, numbers = self.refine_func(dataset)
        print(f'Implemented {self.refiner_name}. {numbers} data refined.', flush=True)
        
        return refined_dataset
