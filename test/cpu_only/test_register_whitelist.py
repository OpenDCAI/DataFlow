from dataflow.utils.registry import OPERATOR_REGISTRY
from pprint import pprint

if __name__ == "__main__":
    # Get all registered operators
    OPERATOR_REGISTRY._get_all()
    print(OPERATOR_REGISTRY)
    # pprint(OPERATOR_REGISTRY.get_type_of_objects())

    # Apply whitelist
    OPERATOR_REGISTRY.apply_whitelist(["PromptedGenerator"])
    dataflow_obj_map = OPERATOR_REGISTRY.get_obj_map()
    # Only one operator should remain
    print("After applying whitelist:")
    print(OPERATOR_REGISTRY)
