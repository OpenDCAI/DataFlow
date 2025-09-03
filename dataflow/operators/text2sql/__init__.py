from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # filter
    from filter.sql_consistency_filter import SQLConsistencyFilter
    from filter.sql_execution_filter import SQLExecutionFilter
    from filter.vecsql_execution_filter import VecSQLExecutionFilter

    # generate
    from generate.sql_generator import SQLGenerator
    from generate.vecsql_generator import VecSQLGenerator
    from generate.sql_variation_generator import SQLVariationGenerator
    from generate.text2sql_cot_generator import Text2SQLCoTGenerator
    from generate.text2sql_prompt_generator import Text2SQLPromptGenerator
    from generate.text2vecsql_prompt_generator import Text2VecSQLPromptGenerator
    from generate.text2sql_question_generator import Text2SQLQuestionGenerator
    from generate.text2vecsql_question_generator import Text2VecSQLQuestionGenerator

    # eval
    from eval.sql_component_classifier import SQLComponentClassifier
    from eval.sql_execution_classifier import SQLExecutionClassifier

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking



    cur_path = "dataflow/operators/text2sql/"


    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/text2sql/", _import_structure)
