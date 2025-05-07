from tqdm import tqdm
from nltk import word_tokenize
import logging
import os
import pandas as pd
import sqlite3
import re
from dataflow.utils.registry import GENERATOR_REGISTRY
from dataflow.utils.utils import get_logger


class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema):
        self._schema = self._normalize_schema(schema)
        self._idMap = self._map(self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _normalize_schema(self, schema):
        # 所有表名和列名做 lower 和 strip 处理
        normalized = {}
        for table, cols in schema.items():
            table_clean = table.strip().lower()
            normalized[table_clean] = [col.strip().lower() for col in cols]
        return normalized

    def _map(self, schema):
        idMap = {'*': "__all__"}
        for table, cols in schema.items():
            for col in cols:
                full = f"{table}.{col}"
                idMap[full] = f"__{full}__"
        for table in schema:
            idMap[table] = f"__{table}__"
        return idMap
    

class EvalHardness:
    """
    Tokenize SQL query and parse SQL query
    """
    CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
    JOIN_KEYWORDS = ('join', 'on', 'as')

    WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
    UNIT_OPS = ('none', '-', '+', "*", '/')
    AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
    TABLE_TYPE = {
        'sql': "sql",
        'table_unit': "table_unit",
    }

    COND_OPS = ('and', 'or')
    SQL_OPS = ('intersect', 'union', 'except')
    ORDER_OPS = ('desc', 'asc')
    FUNC_OPS = ('cast', 'substring', 'date', 'round', 'coalesce')
    def __init__(self, schema, query):
        self.schema = schema
        self.query = query

    @property
    def tokenize(self):
        string = str(self.query)
        vals = {}

        # 替换被 '、"、` 包裹的内容，标记类型
        def replace_literal(match):
            val = match.group(0)
            quote = val[0]
            content = val[1:-1]  # 去除引号

            if quote == '`':
                key = f"__col_{len(vals)}__"
            else:  # ' 或 "
                key = f"__str_{len(vals)}__"

            vals[key] = content
            return key

        # 正则提取引号内容
        string = re.sub(r"(['\"`])(?:\\.|[^\\])*?\1", replace_literal, string)

        toks = [word.lower() for word in word_tokenize(string)]
        # replace with string value token
        for i in range(len(toks)):
            if toks[i] in vals:
                toks[i] = vals[toks[i]]

        # find if there exists !=, >=, <=
        eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
        eq_idxs.reverse()
        prefix = ('!', '>', '<')
        for eq_idx in eq_idxs:
            pre_tok = toks[eq_idx-1]
            if pre_tok in prefix:
                toks = toks[:eq_idx-1] + [pre_tok + "="] + toks[eq_idx+1: ]

        return toks  
    
    def scan_alias(self, toks):
        """Scan the index of 'as' and build the map for all alias"""
        as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
        alias = {}
        for idx in as_idxs:
            alias[toks[idx+1]] = toks[idx-1]
        return alias

    def get_tables_with_alias(self, toks):
        tables = self.scan_alias(toks)
        for key in self.schema.schema:
            assert key not in tables, "Alias {} has the same name in table".format(key)
            tables[key] = key
        return tables
    
    def parse_col(self, toks, start_idx, tables_with_alias, schema, default_tables=None):
        """
            :returns next idx, column id
        """
        tok = toks[start_idx].strip().lower()
        if tok == "*":
            return start_idx + 1, schema.idMap[tok]

        if '.' in tok:  # if token is a composite
            alias, col = tok.split('.')
            key = tables_with_alias[alias] + "." + col
            return start_idx+1, schema.idMap[key]

        assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

        for alias in default_tables:
            table = tables_with_alias[alias]
            if tok in schema.schema[table]:
                key = table + "." + tok
                return start_idx+1, schema.idMap[key]

        assert False, "Error col: {}".format(tok)


    def parse_col_unit(self, toks, start_idx, tables_with_alias, schema, default_tables=None):
        """
            :returns next idx, (agg_op id, col_id)
        """
        idx = start_idx
        if toks[idx] in self.FUNC_OPS:
            func_name = toks[idx]
            idx += 1
            assert toks[idx] == '('
            idx += 1

            if func_name == 'cast':
                # 解析 cast(col as type)
                idx, col_id = self.parse_col(toks, idx, tables_with_alias, schema, default_tables)
                assert toks[idx] == 'as'
                idx += 1
                data_type = toks[idx]
                idx += 1
                assert toks[idx] == ')'
                idx += 1

                func_call = ('func', func_name, col_id, data_type)
                return idx, (self.AGG_OPS.index("none"), func_call, False)

            else:
                # 其他函数比如 round(col, 2)
                idx, col_id = self.parse_col(toks, idx, tables_with_alias, schema, default_tables)
                # 跳过逗号和其他参数（简单版本）
                while toks[idx] != ')':
                    idx += 1
                idx += 1

                func_call = ('func', func_name, col_id, None)
                return idx, (self.AGG_OPS.index("none"), func_call, False)
        len_ = len(toks)
        isBlock = False
        isDistinct = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] in self.AGG_OPS:
            agg_id = self.AGG_OPS.index(toks[idx])
            idx += 1
            assert idx < len_ and toks[idx] == '('
            idx += 1
            if toks[idx] == "distinct":
                idx += 1
                isDistinct = True
            idx, col_id = self.parse_col(toks, idx, tables_with_alias, schema, default_tables)
            assert idx < len_ and toks[idx] == ')'
            idx += 1
            return idx, (agg_id, col_id, isDistinct)

        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        agg_id = self.AGG_OPS.index("none")
        idx, col_id = self.parse_col(toks, idx, tables_with_alias, schema, default_tables)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1  # skip ')'

        return idx, (agg_id, col_id, isDistinct)

    def parse_val_unit(self, toks, start_idx, tables_with_alias, schema, default_tables=None):
        idx = start_idx
        if toks[idx] in self.FUNC_OPS:
            func_name = toks[idx]
            idx += 1
            assert toks[idx] == '('
            idx += 1

            if func_name == 'cast':
                # 解析 cast(col as type)
                idx, col_id = self.parse_col(toks, idx, tables_with_alias, schema, default_tables)
                assert toks[idx] == 'as'
                idx += 1
                data_type = toks[idx]
                idx += 1
                assert toks[idx] == ')'
                idx += 1

                func_call = ('func', func_name, col_id, data_type)
                return idx, (self.AGG_OPS.index("none"), func_call, False)

            else:
                # 其他函数比如 round(col, 2)
                idx, col_id = self.parse_col(toks, idx, tables_with_alias, schema, default_tables)
                # 跳过逗号和其他参数（简单版本）
                while toks[idx] != ')':
                    idx += 1
                idx += 1

                func_call = ('func', func_name, col_id, None)
                return idx, (self.AGG_OPS.index("none"), func_call, False)
        len_ = len(toks)
        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        col_unit1 = None
        col_unit2 = None
        unit_op = self.UNIT_OPS.index('none')

        idx, col_unit1 = self.parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
        if idx < len_ and toks[idx] in self.UNIT_OPS:
            unit_op = self.UNIT_OPS.index(toks[idx])
            idx += 1
            idx, col_unit2 = self.parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1  # skip ')'

        return idx, (unit_op, col_unit1, col_unit2)


    def parse_table_unit(self, toks, start_idx, tables_with_alias, schema):
        """
            :returns next idx, table id, table name
        """
        idx = start_idx
        len_ = len(toks)
        key = tables_with_alias[toks[idx]]

        if idx + 1 < len_ and toks[idx+1] == "as":
            idx += 3
        else:
            idx += 1

        return idx, schema.idMap[key], key


    def parse_value(self, toks, start_idx, tables_with_alias, schema, default_tables=None):
        idx = start_idx
        len_ = len(toks)

        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] == 'select':
            idx, val = self.parse_sql(toks, idx, tables_with_alias, schema)
        elif isinstance(toks[idx], str) and toks[idx] not in schema.idMap:
            # 字符串字面值（非列名）
            val = toks[idx]
            idx += 1
        else:
            try:
                val = float(toks[idx])  # 数字
                idx += 1
            except:
                end_idx = idx
                while end_idx < len_ and toks[end_idx] not in (',', ')', 'and', *self.CLAUSE_KEYWORDS, *self.JOIN_KEYWORDS):
                    end_idx += 1

                idx, val = self.parse_col_unit(toks[start_idx:end_idx], 0, tables_with_alias, schema, default_tables)
                idx = end_idx

        if isBlock:
            assert toks[idx] == ')'
            idx += 1

        return idx, val

    def parse_condition(self, toks, start_idx, tables_with_alias, schema, default_tables=None):
        idx = start_idx
        len_ = len(toks)
        conds = []

        while idx < len_:
            idx, val_unit = self.parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
            not_op = False
            if toks[idx] == 'not':
                not_op = True
                idx += 1

            assert idx < len_ and toks[idx] in self.WHERE_OPS, "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
            op_id = self.WHERE_OPS.index(toks[idx])
            idx += 1
            val1 = val2 = None
            if op_id == self.WHERE_OPS.index('between'):  # between..and... special case: dual values
                idx, val1 = self.parse_value(toks, idx, tables_with_alias, schema, default_tables)
                assert toks[idx] == 'and'
                idx += 1
                idx, val2 = self.parse_value(toks, idx, tables_with_alias, schema, default_tables)
            else:  # normal case: single value
                idx, val1 = self.parse_value(toks, idx, tables_with_alias, schema, default_tables)
                val2 = None

            conds.append((not_op, op_id, val_unit, val1, val2))

            if idx < len_ and (toks[idx] in self.CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in self.JOIN_KEYWORDS):
                break

            if idx < len_ and toks[idx] in self.COND_OPS:
                conds.append(toks[idx])
                idx += 1  # skip and/or

        return idx, conds


    def parse_select(self, toks, start_idx, tables_with_alias, schema, default_tables=None):
        idx = start_idx
        len_ = len(toks)

        assert toks[idx] == 'select', "'select' not found"
        idx += 1
        isDistinct = False
        if idx < len_ and toks[idx] == 'distinct':
            idx += 1
            isDistinct = True
        val_units = []

        while idx < len_ and toks[idx] not in self.CLAUSE_KEYWORDS:
            agg_id = self.AGG_OPS.index("none")
            if toks[idx] in self.AGG_OPS:
                agg_id = self.AGG_OPS.index(toks[idx])
                idx += 1
            idx, val_unit = self.parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
            val_units.append((agg_id, val_unit))
            if idx < len_ and toks[idx] == ',':
                idx += 1  # skip ','

        return idx, (isDistinct, val_units)


    def parse_from(self, toks, start_idx, tables_with_alias, schema):
        """
        Assume in the from clause, all table units are combined with join
        """
        assert 'from' in toks[start_idx:], "'from' not found"

        len_ = len(toks)
        idx = toks.index('from', start_idx) + 1
        default_tables = []
        table_units = []
        conds = []

        while idx < len_:
            isBlock = False
            if toks[idx] == '(':
                isBlock = True
                idx += 1

            if toks[idx] == 'select':
                idx, sql = self.parse_sql(toks, idx, tables_with_alias, schema)
                table_units.append((self.TABLE_TYPE['sql'], sql))
            else:
                if idx < len_ and toks[idx] == 'join':
                    idx += 1  # skip join
                idx, table_unit, table_name = self.parse_table_unit(toks, idx, tables_with_alias, schema)
                table_units.append((self.TABLE_TYPE['table_unit'],table_unit))
                default_tables.append(table_name)
            if idx < len_ and toks[idx] == "on":
                idx += 1  # skip on
                idx, this_conds = self.parse_condition(toks, idx, tables_with_alias, schema, default_tables)
                if len(conds) > 0:
                    conds.append('and')
                conds.extend(this_conds)

            if isBlock:
                assert toks[idx] == ')'
                idx += 1
            if idx < len_ and (toks[idx] in self.CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
                break

        return idx, table_units, conds, default_tables


    def parse_where(self, toks, start_idx, tables_with_alias, schema, default_tables):
        idx = start_idx
        len_ = len(toks)

        if idx >= len_ or toks[idx] != 'where':
            return idx, []

        idx += 1
        idx, conds = self.parse_condition(toks, idx, tables_with_alias, schema, default_tables)
        return idx, conds


    def parse_group_by(self, toks, start_idx, tables_with_alias, schema, default_tables):
        idx = start_idx
        len_ = len(toks)
        col_units = []

        if idx >= len_ or toks[idx] != 'group':
            return idx, col_units

        idx += 1
        assert toks[idx] == 'by'
        idx += 1

        while idx < len_ and not (toks[idx] in self.CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            idx, col_unit = self.parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
            col_units.append(col_unit)
            if idx < len_ and toks[idx] == ',':
                idx += 1  # skip ','
            else:
                break

        return idx, col_units


    def parse_order_by(self, toks, start_idx, tables_with_alias, schema, default_tables):
        idx = start_idx
        len_ = len(toks)
        val_units = []
        order_type = 'asc' # default type is 'asc'

        if idx >= len_ or toks[idx] != 'order':
            return idx, val_units

        idx += 1
        assert toks[idx] == 'by'
        idx += 1

        while idx < len_ and not (toks[idx] in self.CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            idx, val_unit = self.parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
            val_units.append(val_unit)
            if idx < len_ and toks[idx] in self.ORDER_OPS:
                order_type = toks[idx]
                idx += 1
            if idx < len_ and toks[idx] == ',':
                idx += 1  # skip ','
            else:
                break

        return idx, (order_type, val_units)


    def parse_having(self, toks, start_idx, tables_with_alias, schema, default_tables):
        idx = start_idx
        len_ = len(toks)

        if idx >= len_ or toks[idx] != 'having':
            return idx, []

        idx += 1
        idx, conds = self.parse_condition(toks, idx, tables_with_alias, schema, default_tables)
        return idx, conds


    def parse_limit(self, toks, start_idx):
        idx = start_idx
        len_ = len(toks)

        if idx < len_ and toks[idx] == 'limit':
            idx += 2
            # make limit value can work, cannot assume put 1 as a fake limit number
            if type(toks[idx-1]) != int:
                return idx, 1

            return idx, int(toks[idx-1])

        return idx, None

    def skip_semicolon(self, toks, start_idx):
        idx = start_idx
        while idx < len(toks) and toks[idx] == ";":
            idx += 1
        return idx

    def parse_sql(self, start_idx):
        toks = self.tokenize
        tables_with_alias = self.get_tables_with_alias(toks)
        schema =self.schema
        isBlock = False # indicate whether this is a block of sql/sub-sql
        len_ = len(toks)
        idx = start_idx

        sql = {}
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        # parse from clause in order to get default tables
        from_end_idx, table_units, conds, default_tables = self.parse_from(toks, start_idx, tables_with_alias, schema)
        sql['from'] = {'table_units': table_units, 'conds': conds}
        # select clause
        _, select_col_units = self.parse_select(toks, idx, tables_with_alias, schema, default_tables)
        idx = from_end_idx
        sql['select'] = select_col_units
        # where clause
        idx, where_conds = self.parse_where(toks, idx, tables_with_alias, schema, default_tables)
        sql['where'] = where_conds
        # group by clause
        idx, group_col_units = self.parse_group_by(toks, idx, tables_with_alias, schema, default_tables)
        sql['groupBy'] = group_col_units
        # having clause
        idx, having_conds = self.parse_having(toks, idx, tables_with_alias, schema, default_tables)
        sql['having'] = having_conds
        # order by clause
        idx, order_col_units = self.parse_order_by(toks, idx, tables_with_alias, schema, default_tables)
        sql['orderBy'] = order_col_units
        # limit clause
        idx, limit_val = self.parse_limit(toks, idx)
        sql['limit'] = limit_val

        idx = self.skip_semicolon(toks, idx)
        if isBlock:
            assert toks[idx] == ')'
            idx += 1  # skip ')'
        idx = self.skip_semicolon(toks, idx)

        # intersect/union/except clause
        for op in self.SQL_OPS:  # initialize IUE
            sql[op] = None
        if idx < len_ and toks[idx] in self.SQL_OPS:
            sql_op = toks[idx]
            idx += 1
            idx, IUE_sql = self.parse_sql(idx)
            sql[sql_op] = IUE_sql
        return idx, sql
    
    def has_agg(self, unit):
        return unit[0] != self.AGG_OPS.index('none')
    
    def count_agg(self, units):
        return len([unit for unit in units if self.has_agg(unit)])
    
    def get_nestedSQL(self, sql):
        nested = []
        for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
            if type(cond_unit[3]) is dict:
                nested.append(cond_unit[3])
            if type(cond_unit[4]) is dict:
                nested.append(cond_unit[4])
        if sql['intersect'] is not None:
            nested.append(sql['intersect'])
        if sql['except'] is not None:
            nested.append(sql['except'])
        if sql['union'] is not None:
            nested.append(sql['union'])
        return nested

    def count_component1(self, sql):
        count = 0
        if len(sql['where']) > 0:
            count += 1
        if len(sql['groupBy']) > 0:
            count += 1
        if len(sql['orderBy']) > 0:
            count += 1
        if sql['limit'] is not None:
            count += 1
        if len(sql['from']['table_units']) > 0:  # JOIN
            count += len(sql['from']['table_units']) - 1

        ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
        count += len([token for token in ao if token == 'or'])
        cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
        count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == self.WHERE_OPS.index('like')])

        return count


    def count_component2(self, sql):
        nested = self.get_nestedSQL(sql)
        return len(nested)


    def count_others(self, sql):
        count = 0
        # number of aggregation
        agg_count = self.count_agg(sql['select'][1])
        agg_count += self.count_agg(sql['where'][::2])
        agg_count += self.count_agg(sql['groupBy'])
        if len(sql['orderBy']) > 0:
            agg_count += self.count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                                [unit[2] for unit in sql['orderBy'][1] if unit[2]])
        agg_count += self.count_agg(sql['having'])
        if agg_count > 1:
            count += 1

        # number of select columns
        if len(sql['select'][1]) > 1:
            count += 1

        # number of where conditions
        if len(sql['where']) > 1:
            count += 1

        # number of group by clauses
        if len(sql['groupBy']) > 1:
            count += 1

        return count

        
    def eval_hardness(self, sql):
        count_comp1_ = self.count_component1(sql)
        count_comp2_ = self.count_component2(sql)
        count_others_ = self.count_others(sql)

        if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
            return "easy"
        elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
                (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
            return "medium"
        elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
                (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
                (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
            return "hard"
        else:
            return "extra"
        
    def run(self):
        _, sql = self.parse_sql(0)
        hardness = self.eval_hardness(sql)
        return hardness

import re

class EvalHardnessLite:
    def __init__(self, sql: str):
        self.sql = sql.lower()

    def match(self, pattern):
        return bool(re.search(pattern, self.sql))

    def count_keyword(self, keyword):
        return self.sql.count(keyword)

    def run(self):
        sql = self.sql
        score = 0

        # 子查询检测
        if self.match(r'\( *select'):
            score += 2

        # 多表 JOIN 检测
        if self.count_keyword(' join ') > 0:
            score += self.count_keyword(' join ')
        if self.count_keyword(',') > 0 and 'from' in sql:
            score += 1  # 暂定认为逗号连接表增加复杂度

        # 条件检测
        if self.count_keyword(' and ') + self.count_keyword(' or ') >= 2:
            score += 1
        if any(kw in sql for kw in ['in', 'exists', 'like']):
            score += 1

        # 分组和聚合
        if 'group by' in sql:
            score += 1
        if 'having' in sql:
            score += 1

        # 函数调用
        if any(func in sql for func in ['cast', 'round', 'substring', 'date', 'coalesce']):
            score += 1

        # 排序 + 限制
        if 'order by' in sql:
            score += 1
        if 'limit' in sql:
            score += 1

        # set 操作
        if any(op in sql for op in ['union', 'intersect', 'except']):
            score += 2

        # 多列 SELECT
        select_cols = re.findall(r'select\s+(distinct\s+)?(.+?)\s+from', sql, re.DOTALL)
        if select_cols:
            num_commas = select_cols[0][1].count(',')
            if num_commas >= 1:
                score += 1

        # 简单规则分类
        if score <= 2:
            return 'easy'
        elif score <= 4:
            return 'medium'
        elif score <= 6:
            return 'hard'
        else:
            return 'extra'

@GENERATOR_REGISTRY.register()
class SQLDifficultyClassifier:
    def __init__(self, config: dict):
        '''
        Initialize the SQLDifficultyClassifier with the provided configuration.
        '''
        self.config = config

        # Input and output file paths and keys
        self.input_file = self.config.get("input_file")
        self.output_file = self.config.get("output_file")
        self.input_sql_key = self.config.get("input_sql_key", "SQL")
        # self.input_dbid_key = self.config.get("input_dbid_key", "db_id")
        self.output_key = self.config.get("output_key", "sql_difficulty")
        # self.db_root_path = self.config.get("db_root_path")

        self.logger = get_logger()

        # Ensure required paths and keys are provided
        if not self.input_file or not self.output_file:
            self.logger.error("Both input_file and output_file must be specified in the config.")
            raise ValueError("Both input_file and output_file must be specified in the config.")
        
    def get_schema(self, db):
        """
        Get database's schema, which is a dict with table name as key
        and list of column names as value
        :param db: database path
        :return: schema dict
        """

        schema = {}
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        # fetch table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [str(table[0].lower()) for table in cursor.fetchall()]

        # fetch table info
        for table in tables:
            cursor.execute("PRAGMA table_info({})".format(table))
            schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]

        return schema    
    
    def report_statistics(self, dataframe: pd.DataFrame):
        '''
        print the statistics of the SQL difficulty.
        '''
        counts = dataframe[self.output_key].value_counts()
        self.logger.info("SQL Difficulty Statistics")
        difficulty_counts = {d: counts.get(d, 0) for d in ['easy', 'medium', 'hard', 'extra']}
        self.logger.info(" | ".join([f"{d.title()}: {v}" for d, v in difficulty_counts.items()]))
    
    def run(self):
        '''
        Runs the TextSQLDifficultyClassifier, reading from the input file and saving results to output.
        '''
        # Read input file: only accept jsonl format
        dataframe = pd.read_json(self.input_file, lines=True)
        
        # Ensure the input and output keys are correctly set
        self._validate_dataframe(dataframe)

        # Process each row in the DataFrame
        # for idx, row in dataframe.iterrows():
        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Processing"):
            # db = os.path.join(self.db_root_path, row[self.input_dbid_key], row[self.input_dbid_key] + ".sqlite")
            # schema = Schema(self.get_schema(db))
            sql = row[self.input_sql_key]
            # sql_hardness = EvalHardness(schema, sql)
            sql_hardness = EvalHardnessLite(sql)
            hardness = sql_hardness.run()
            dataframe.at[idx, self.output_key] = hardness

        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)

        # Save DataFrame to JSON file
        self.report_statistics(dataframe)
        dataframe.to_json(self.output_file, orient="records", lines=True, force_ascii=False)

        
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        '''
        Helper method to validate the input dataframe columns.
        '''
        # Check if the input sql key exists in the dataframe
        if self.input_sql_key not in dataframe.columns:
            self.logger.error(f"input_sql_key: {self.input_sql_key} not found in the dataframe.")
            raise ValueError(f"input_sql_key: {self.input_sql_key} not found in the dataframe.")
        
        # Check if the input dbid key exists in the dataframe
        # if self.input_dbid_key not in dataframe.columns:
        #     raise ValueError(f"input_dbid_key: {self.input_dbid_key} not found in the dataframe.")

        # Check if the output key already exists in the dataframe
        if self.output_key in dataframe.columns:
            self.logger.warning(f"Found {self.output_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")
            raise ValueError(f"Found {self.output_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")
