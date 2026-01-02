import os
from pathlib import Path
import appdirs

class DataFlowPath:
    """
    Class to manage paths for DataFlow.
    """


    @staticmethod
    def get_dataflow_dir():
        # return path of /dataflow
        return Path(__file__).parent.parent 

    # @staticmethod
    # def get_dataflow_scripts_dir():
        # return DataFlowPath.get_dataflow_dir() / "scripts"

    @staticmethod
    def get_dataflow_example_dir():
        return DataFlowPath.get_dataflow_dir() / "example"
    
    @staticmethod
    def get_dataflow_statics_dir():
        return DataFlowPath.get_dataflow_dir() / "statics"
    
    @staticmethod
    def get_dataflow_pipelines_dir():
        return DataFlowPath.get_dataflow_statics_dir() / "pipelines"
    
    @staticmethod
    def get_dataflow_playground_dir():
        return DataFlowPath.get_dataflow_statics_dir() / "playground"
    
    @staticmethod
    def get_dataflow_scaffold_dir():
        return DataFlowPath.get_dataflow_statics_dir() / "scaffold"