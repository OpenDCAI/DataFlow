'''
A collection of prompts for the text2sql operator.
'''
class TextSQLConsistencyPrompt:
    def __init__(self):
        pass

    def text_sql_consistency_prompt(self, question, sql, schema):
        prompt = f"""
        **Task Overview**
        Determine if the SQL query correctly answers the given question based on the provided schema.

        **Question**
        {question}

        **SQL**
        {sql}

        **Schema**
        {schema}

        **Evaluation Criteria**
        1. **Logical Alignment**: Does the SQL query logically address what the question is asking?
        2. **Schema Compliance**: Are the tables, columns, and relationships used correctly according to the schema?
        3. **Completeness**: Does the SQL capture all necessary conditions and requirements from the question?
        4. **Correctness**: Are there any logical errors that would prevent getting the correct answer?

        **Output Format**:
        The conclusion should be enclosed in a code block:
        ```
        <Conclusion> YES/NO </Conclusion>
        ```

        **Decision Rules**
        - YES: SQL correctly implements the question requirements
        - NO: SQL has logical errors or doesn't address the question properly
        - When uncertain about edge cases, explain the uncertainty in analysis but still provide a definitive YES/NO

        **Answer**
        Let's proceed step by step.
        """
        return prompt

class CotGenerationPrompt:
    def __init__(self):
        pass

    def text2sql_cot_prompt(self, schema, question, sql):
        prompt = f"""
        You are a senior data analyst specializing in SQL. Your task is to translate a natural language question into an executable SQLite query, providing a detailed reasoning trace.

        You will also receive a reference solution from a colleague, which may or may not be correct. This extra information intends to help you generate your answer, but you are asked not to mention the reference solution in any form.
        The reference solution might include: 
        1. Unnecessary table and column selections. 
        2. Incorrect or excessive joins. 
        3. Misalignment with the question.
        4. Opportunities for simplification.

        Ensure the SQL query is presented in a Markdown code block with proper syntax highlighting, like this:
        ```sql
        SELECT * FROM table;
        ```

        [Database Schema]:
        {schema}

        [Natural Language Question]:
        {question}

        [Reference Solution]:
        ```sql
        {sql}
        ```

        Provide your step-by-step text-to-SQL solution here.
        """
        return prompt


class SQLGenerationPrompt:
    def __init__(self):
        self.simple_criterion = '''**Criteria:**
        Simple SQL queries may satisfy one or more of the following criteria:
        - Simple queries should select data from a single table only.
        - Basic aggregate functions are permitted, such as `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`.
        - No joins are allowed; the query must operate on a single table.

        **Example of Simple SQL Query:**
        ```sql
        SELECT name, department_name
        FROM employees
        WHERE level > 5
        ORDER BY age DESC;
        ```'''
    
        self.moderate_criterion = '''**Criteria:**
        Moderate SQL queries may satisfy one or more of the following criteria:
        - Involves table joins, such as `JOIN`, `INNER JOIN`, `LEFT JOIN`, `CROSS JOIN`, etc.
        - Includes subqueries within the `SELECT` or `WHERE` clauses.
        - Utilizes aggregate functions alongside a `GROUP BY` clause.
        - Contains complex `WHERE` conditions, including `IN`, `BETWEEN`, `LIKE`.
        - Incorporate a `HAVING` clause to filter aggregated results.
        - Uses aggregate functions like `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, etc.

        **Example of Moderate SQL Query:**
        ```sql
        SELECT e.name, d.department_name, AVG(s.salary) AS average_salary
        FROM employees e
        INNER JOIN departments d ON e.department_id = d.department_id
        LEFT JOIN salaries s ON e.employee_id = s.employee_id
        WHERE e.age > 30 AND e.status = 'active'
        GROUP BY e.name, d.department_name
        HAVING AVG(s.salary) > 50000;
        ```'''

        self.complex_criterion = '''**Criteria:**
        Complex SQL queries may satisfy one or more of the following criteria:
        - Contains complex nested subqueries.
        - Utilizes multiple types of joins, including self-joins.
        - Includes window functions, such as `ROW_NUMBER`, `RANK`, etc.
        - Uses Common Table Expressions (CTEs) for improved readability.
        - Combines multiple aggregate functions.
        - Involves complex `WHERE` and `HAVING` clauses with multiple conditions.
        - Utilizes advanced functions and operators.

        **Example of Complex SQL Query:**
        ```sql
        WITH EmployeeCTE AS (
            SELECT employee_id, name, department_id, ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rank
            FROM employees
        )
        SELECT e.name, d.department_name
        FROM EmployeeCTE e
        INNER JOIN departments d ON e.department_id = d.department_id
        WHERE e.rank <= 3;
        ```'''

        self.highly_complex_criterion = '''**Criteria:**
        Highly complex SQL queries may satisfy one or more of the following criteria:
        - Includes multiple Common Table Expressions (CTEs) for readability.
        - Combines nested subqueries and various joins.
        - Utilizes recursive CTEs for hierarchical or recursive queries.
        - Extensively uses advanced window functions.
        - May involve `UNION` or `UNION ALL` to combine result sets.
        - Implements complex logic with advanced analytical functions.
        - Employs a wide range of SQL clauses and conditions.
        - Utilizes a broad spectrum of SQL functions and advanced features.

        **Example of Highly Complex SQL Query:**
        ```sql
        WITH RECURSIVE EmployeeHierarchy AS (
            SELECT employee_id, name, manager_id, department_id, 1 as level
            FROM employees
            WHERE manager_id IS NULL
            UNION ALL
            SELECT e.employee_id, e.name, e.manager_id, e.department_id, eh.level + 1
            FROM employees e
            JOIN EmployeeHierarchy eh ON e.manager_id = eh.employee_id
        ),
        DepartmentSalaries AS (
            SELECT eh.employee_id, eh.name, eh.level, d.department_name, s.salary, d.department_id
            FROM EmployeeHierarchy eh
            INNER JOIN departments d ON eh.department_id = d.department_id
            INNER JOIN salaries s ON eh.employee_id = s.employee_id
        ),
        DepartmentStats AS (
            SELECT 
                d.department_id,
                COUNT(e.employee_id) AS employee_count,
                AVG(s.salary) AS average_salary
            FROM employees e
            INNER JOIN salaries s ON e.employee_id = s.employee_id
            INNER JOIN departments d ON e.department_id = d.department_id
            GROUP BY d.department_id
        )
        SELECT ds.name, ds.level, 
            SUM(ds.salary) OVER (PARTITION BY ds.department_id ORDER BY ds.level, ds.name) AS cumulative_salary
        FROM DepartmentSalaries ds
        INNER JOIN DepartmentStats dstat ON ds.department_id = dstat.department_id
        ORDER BY ds.level, ds.name;
        ```'''

    def sql_func_template(self, sql_funcs):
        template = """### SQL Functions
        You may consider one or more of the following SQL functions while generating the query:
        {sql_funcs}
        Important tips:
        Except for the functions listed above, you may use any other functions as long as they conform to the syntax of the database engine.
        """
        return template.format(sql_funcs=sql_funcs)

    def insert_stmts_template(self, insert_statements):
        template = '''### INSERT INTO Statements
        Below are several `INSERT INTO` statements. Use these to help generate predicates (i.e., `WHERE` clauses) in your SQL query:
        {insert_statements}
        '''
        return template.format(insert_statements=insert_statements)

    def sql_synthesis_prompt(self, schema_str, sql_function_prompt, db_value_prompt, complexity, criterion, db_engine, column_count):
        template = '''**Task Overview**
        Create an executable SQL query based on the provided information.

        **Database Schema**
        {schema_str}

        {sql_function_prompt}

        {db_value_prompt}

        **SQL Query Complexity**
        Ensure the SQL query matches the {complexity} level, defined as follows:
        {criterion}

        **Output Format Requirements**
        Enclose the SQL query in a code block:
        ```sql
        -- Your SQL query here
        ```

        **SQL Query Requirements**
        1. Use the syntax specific to the {db_engine} database engine.
        2. Incorporate advanced functions if appropriate, but they are not mandatory.
        3. Address real-world data analysis needs. Avoid trivial or nonsensical queries.
        4. (Very important) Ensure the final SQL query selects {column_count} columns.

        **Answer**
        Let's proceed step by step.
        '''
        return template.format(
            schema_str=schema_str,
            sql_function_prompt=sql_function_prompt.strip(),
            db_value_prompt=db_value_prompt.strip(),
            complexity=complexity,
            criterion=criterion.strip(),
            db_engine=db_engine,
            column_count=column_count
        )
    
    def sqlite_funcs(self):
        funcs = [
            "ABS(X) \nDescription: The ABS(X) function returns the absolute value of the numeric argument X. Abs(X) returns NULL if X is NULL. Abs(X) returns 0.0 if X is a string or blob that cannot be converted to a numeric value. If X is the integer -9223372036854775808 then ABS(X) throws an integer overflow error since there is no equivalent positive 64-bit two complement value. ",
            "CHANGES() \nDescription: The CHANGES() function returns the number of database rows that were changed or inserted or deleted by the most recently completed INSERT, DELETE, or UPDATE statement, exclusive of statements in lower-level triggers. The CHANGES() SQL function is a wrapper around thesqlite3_changes64()C/C++ function and hence follows the same rules for counting changes. ",
            "CHAR(X1,X2,...,XN) \nDescription: The CHAR(X1,X2,...,XN) function returns a string composed of characters having the unicode code point values of integers X1 through XN, respectively. ",
            "COALESCE(X,Y,...) \nDescription: The COALESCE() function returns a copy of its first non-NULL argument, or NULL if all arguments are NULL. Coalesce() must have at least 2 arguments. ",
            "CONCAT(X,...) \nDescription: The CONCAT(...) function returns a string which is the concatenation of the string representation of all of its non-NULL arguments. If all arguments are NULL, then CONCAT() returns an empty string. ",
            "CONCAT_WS(SEP,X,...) \nDescription: The CONCAT_WS(SEP,...) function returns a string that is the concatenation of all non-null arguments beyond the first argument, using the text value of the first argument as a separator. If the first argument is NULL, then CONCAT_WS() returns NULL. If all arguments other than the first are NULL, then CONCAT_WS() returns an empty string. ",
            "FORMAT(FORMAT,...) \nDescription: The FORMAT(FORMAT,...) SQL function works like thesqlite3_mprintf()C-language function and the printf() function from the standard C library. The first argument is a format string that specifies how to construct the output string using values taken from subsequent arguments. If the FORMAT argument is missing or NULL then the result is NULL. The %n format is silently ignored and does not consume an argument. The %p format is an alias for %X. The %z format is interchangeable with %s. If there are too few arguments in the argument list, missing arguments are assumed to have a NULL value, which is translated into 0 or 0.0 for numeric formats or an empty string for %s. See thebuilt-in printf()documentation for additional information. ",
            "GLOB(X,Y) \nDescription: The GLOB(X,Y) function is equivalent to the expression \"Y GLOB X\". Note that the X and Y arguments are reversed in the GLOB() function relative to the infixGLOBoperator. Y is the string and X is the pattern. So, for example, the following expressions are equivalent:name GLOB '*helium*' GLOB('*helium*',name)If thesqlite3_create_function()interface is used to override the GLOB(X,Y) function with an alternative implementation then theGLOBoperator will invoke the alternative implementation. ",
            "HEX(X) \nDescription: The HEX() function interprets its argument as a BLOB and returns a string which is the upper-case hexadecimal rendering of the content of that blob.If the argumentXin \"hex(X)\" is an integer or floating point number, then \"interprets its argument as a BLOB\" means that the binary number is first converted into a UTF8 text representation, then that text is interpreted as a BLOB. Hence, \"hex(12345678)\" renders as \"3132333435363738\" not the binary representation of the integer value \"0000000000BC614E\".See also:unhex() ",
            "IFNULL(X,Y) \nDescription: The IFNULL() function returns a copy of its first non-NULL argument, or NULL if both arguments are NULL. Ifnull() must have exactly 2 arguments. The IFNULL() function is equivalent tocoalesce()with two arguments. ",
            "IIF(X,Y,Z) \nDescription: The IIF(X,Y,Z) function returns the value Y if X is true, and Z otherwise. The IIF(X,Y,Z) function is logically equivalent to and generates the samebytecodeas theCASE expression\"CASE WHEN X THEN Y ELSE Z END\". ",
            "INSTR(X,Y) \nDescription: The INSTR(X,Y) function finds the first occurrence of string Y within string X and returns the number of prior characters plus 1, or 0 if Y is nowhere found within X. Or, if X and Y are both BLOBs, then INSTR(X,Y) returns one more than the number bytes prior to the first occurrence of Y, or 0 if Y does not occur anywhere within X. If both arguments X and Y to INSTR(X,Y) are non-NULL and are not BLOBs then both are interpreted as strings. If either X or Y are NULL in INSTR(X,Y) then the result is NULL. ",
            "LAST_INSERT_ROWID() \nDescription: The LAST_INSERT_ROWID() function returns theROWIDof the last row insert from the database connection which invoked the function. The LAST_INSERT_ROWID() SQL function is a wrapper around thesqlite3_last_insert_rowid()C/C++ interface function. ",
            "LENGTH(X) \nDescription: For a string value X, the LENGTH(X) function returns the number of characters (not bytes) in X prior to the first NUL character. Since SQLite strings do not normally contain NUL characters, the LENGTH(X) function will usually return the total number of characters in the string X. For a blob value X, LENGTH(X) returns the number of bytes in the blob. If X is NULL then LENGTH(X) is NULL. If X is numeric then LENGTH(X) returns the length of a string representation of X.Note that for strings, the LENGTH(X) function returns thecharacterlength of the string, not the byte length. The character length is the number of characters in the string. The character length is always different from the byte length for UTF-16 strings, and can be different from the byte length for UTF-8 strings if the string contains multi-byte characters. Use theoctet_length()function to find the byte length of a string.For BLOB values, LENGTH(X) always returns the byte-length of the BLOB.For string values, LENGTH(X) must read the entire string into memory in order to compute the character length. But for BLOB values, that is not necessary as SQLite knows how many bytes are in the BLOB. Hence, for multi-megabyte values, the LENGTH(X) function is usually much faster for BLOBs than for strings, since it does not need to load the value into memory. ",
            "LIKE(X,Y) or LIKE(X,Y,Z) \nDescription: The LIKE() function is used to implement the \"Y LIKE X [ESCAPE Z]\" expression. If the optional ESCAPE clause is present, then the LIKE() function is invoked with three arguments. Otherwise, it is invoked with two arguments only. Note that the X and Y parameters are reversed in the LIKE() function relative to the infixLIKEoperator. X is the pattern and Y is the string to match against that pattern. Hence, the following expressions are equivalent:name LIKE '%neon%' LIKE('%neon%',name)Thesqlite3_create_function()interface can be used to override the LIKE() function and thereby change the operation of theLIKEoperator. When overriding the LIKE() function, it may be important to override both the two and three argument versions of the LIKE() function. Otherwise, different code may be called to implement theLIKEoperator depending on whether or not an ESCAPE clause was specified. ",
            "LIKELIHOOD(X,Y) \nDescription: The LIKELIHOOD(X,Y) function returns argument X unchanged. The value Y in LIKELIHOOD(X,Y) must be a floating point constant between 0.0 and 1.0, inclusive. The LIKELIHOOD(X) function is a no-op that the code generator optimizes away so that it consumes no CPU cycles during run-time (that is, during calls tosqlite3_step()). The purpose of the LIKELIHOOD(X,Y) function is to provide a hint to the query planner that the argument X is a boolean that is true with a probability of approximately Y. Theunlikely(X)function is short-hand for LIKELIHOOD(X,0.0625). Thelikely(X)function is short-hand for LIKELIHOOD(X,0.9375). ",
            "LIKELY(X) \nDescription: The LIKELY(X) function returns the argument X unchanged. The LIKELY(X) function is a no-op that the code generator optimizes away so that it consumes no CPU cycles at run-time (that is, during calls tosqlite3_step()). The purpose of the LIKELY(X) function is to provide a hint to the query planner that the argument X is a boolean value that is usually true. The LIKELY(X) function is equivalent tolikelihood(X,0.9375). See also:unlikely(X). ",
            "LOAD_EXTENSION(X) or LOAD_EXTENSION(X,Y) \nDescription: The LOAD_EXTENSION(X,Y) function loadsSQLite extensionsout of the shared library file named X using the entry point Y. The result of LOAD_EXTENSION() is always a NULL. If Y is omitted then the default entry point name is used. The LOAD_EXTENSION() function raises an exception if the extension fails to load or initialize correctly.The LOAD_EXTENSION() function will fail if the extension attempts to modify or delete an SQL function or collating sequence. The extension can add new functions or collating sequences, but cannot modify or delete existing functions or collating sequences because those functions and/or collating sequences might be used elsewhere in the currently running SQL statement. To load an extension that changes or deletes functions or collating sequences, use thesqlite3_load_extension()C-language API.For security reasons, extension loading is disabled by default and must be enabled by a prior call tosqlite3_enable_load_extension(). ",
            "LOWER(X) \nDescription: The LOWER(X) function returns a copy of string X with all ASCII characters converted to lower case. The default built-in LOWER() function works for ASCII characters only. To do case conversions on non-ASCII characters, load the ICU extension. ",
            "LTRIM(X) or LTRIM(X,Y) \nDescription: The LTRIM(X,Y) function returns a string formed by removing any and all characters that appear in Y from the left side of X. If the Y argument is omitted, LTRIM(X) removes spaces from the left side of X. ",
            "MAX(X,Y,...) \nDescription: The multi-argument MAX() function returns the argument with the maximum value, or return NULL if any argument is NULL. The multi-argument MAX() function searches its arguments from left to right for an argument that defines a collating function and uses that collating function for all string comparisons. If none of the arguments to MAX() define a collating function, then the BINARY collating function is used. Note thatmax()is a simple function when it has 2 or more arguments but operates as anaggregate functionif given only a single argument. ",
            "MIN(X,Y,...) \nDescription: The multi-argument MIN() function returns the argument with the minimum value. The multi-argument MIN() function searches its arguments from left to right for an argument that defines a collating function and uses that collating function for all string comparisons. If none of the arguments to MIN() define a collating function, then the BINARY collating function is used. Note thatmin()is a simple function when it has 2 or more arguments but operates as anaggregate functionif given only a single argument. ",
            "NULLIF(X,Y) \nDescription: The NULLIF(X,Y) function returns its first argument if the arguments are different and NULL if the arguments are the same. The NULLIF(X,Y) function searches its arguments from left to right for an argument that defines a collating function and uses that collating function for all string comparisons. If neither argument to NULLIF() defines a collating function then the BINARY collating function is used. ",
            "OCTET_LENGTH(X) \nDescription: The OCTET_LENGTH(X) function returns the number of bytes in the encoding of text string X. If X is NULL then OCTET_LENGTH(X) returns NULL. If X is a BLOB value, then OCTET_LENGTH(X) is the same aslength(X). If X is a numeric value, then OCTET_LENGTH(X) returns the number of bytes in a text rendering of that number.Because OCTET_LENGTH(X) returns the number of bytes in X, not the number of characters, the value returned depends on the database encoding. The OCTET_LENGTH() function can return different answers for the same input string if the database encoding is UTF16 instead of UTF8.If argument X is a table column and the value is of type text or blob, then OCTET_LENGTH(X) avoids reading the content of X from disk, as the byte length can be computed from metadata. Thus, OCTET_LENGTH(X) is efficient even if X is a column containing a multi-megabyte text or blob value. ",
            "PRINTF(FORMAT,...) \nDescription: The PRINTF() SQL function is an alias for theformat() SQL function. The format() SQL function was originally named PRINTF(). But the name was later changed to format() for compatibility with other database engines. The PRINTF() name is retained as an alias so as not to break legacy code. ",
            "QUOTE(X) \nDescription: The QUOTE(X) function returns the text of an SQL literal which is the value of its argument suitable for inclusion into an SQL statement. Strings are surrounded by single-quotes with escapes on interior quotes as needed. BLOBs are encoded as hexadecimal literals. Strings with embedded NUL characters cannot be represented as string literals in SQL and hence the returned string literal is truncated prior to the first NUL. ",
            "RANDOM() \nDescription: The RANDOM() function returns a pseudo-random integer between -9223372036854775808 and +9223372036854775807. ",
            "RANDOMBLOB(N) \nDescription: The RANDOMBLOB(N) function return an N-byte blob containing pseudo-random bytes. If N is less than 1 then a 1-byte random blob is returned.Hint: applications can generate globally unique identifiers using this function together withhex()and/orlower()like this:hex(randomblob(16))lower(hex(randomblob(16))) ",
            "REPLACE(X,Y,Z) \nDescription: The REPLACE(X,Y,Z) function returns a string formed by substituting string Z for every occurrence of string Y in string X. TheBINARYcollating sequence is used for comparisons. If Y is an empty string then return X unchanged. If Z is not initially a string, it is cast to a UTF-8 string prior to processing. ",
            "ROUND(X) or ROUND(X,Y) \nDescription: The ROUND(X,Y) function returns a floating-point value X rounded to Y digits to the right of the decimal point. If the Y argument is omitted or negative, it is taken to be 0. ",
            "RTRIM(X) or RTRIM(X,Y) \nDescription: The RTRIM(X,Y) function returns a string formed by removing any and all characters that appear in Y from the right side of X. If the Y argument is omitted, RTRIM(X) removes spaces from the right side of X. ",
            "SIGN(X) \nDescription: The SIGN(X) function returns -1, 0, or +1 if the argument X is a numeric value that is negative, zero, or positive, respectively. If the argument to SIGN(X) is NULL or is a string or blob that cannot be losslessly converted into a number, then SIGN(X) returns NULL. ",
            "SOUNDEX(X) \nDescription: The SOUNDEX(X) function returns a string that is the soundex encoding of the string X. The string \"?000\" is returned if the argument is NULL or contains no ASCII alphabetic characters. This function is omitted from SQLite by default. It is only available if theSQLITE_SOUNDEXcompile-time option is used when SQLite is built. ",
            "SQLITE_COMPILEOPTION_GET(N) \nDescription: The SQLITE_COMPILEOPTION_GET() SQL function is a wrapper around thesqlite3_compileoption_get()C/C++ function. This routine returns the N-th compile-time option used to build SQLite or NULL if N is out of range. See also thecompile_options pragma. ",
            "SQLITE_COMPILEOPTION_USED(X) \nDescription: The SQLITE_COMPILEOPTION_USED() SQL function is a wrapper around thesqlite3_compileoption_used()C/C++ function. When the argument X to SQLITE_COMPILEOPTION_USED(X) is a string which is the name of a compile-time option, this routine returns true (1) or false (0) depending on whether or not that option was used during the build. ",
            "SQLITE_OFFSET(X) \nDescription: The SQLITE_OFFSET(X) function returns the byte offset in the database file for the beginning of the record from which value would be read. If X is not a column in an ordinary table, then SQLITE_OFFSET(X) returns NULL. The value returned by SQLITE_OFFSET(X) might reference either the original table or an index, depending on the query. If the value X would normally be extracted from an index, the SQLITE_OFFSET(X) returns the offset to the corresponding index record. If the value X would be extracted from the original table, then SQLITE_OFFSET(X) returns the offset to the table record.The SQLITE_OFFSET(X) SQL function is only available if SQLite is built using the-DSQLITE_ENABLE_OFFSET_SQL_FUNCcompile-time option. ",
            "SQLITE_SOURCE_ID() \nDescription: The SQLITE_SOURCE_ID() function returns a string that identifies the specific version of the source code that was used to build the SQLite library. The string returned by SQLITE_SOURCE_ID() is the date and time that the source code was checked in followed by the SHA3-256 hash for that check-in. This function is an SQL wrapper around thesqlite3_sourceid()C interface. ",
            "SQLITE_VERSION() \nDescription: The SQLITE_VERSION() function returns the version string for the SQLite library that is running. This function is an SQL wrapper around thesqlite3_libversion()C-interface. ",
            "SUBSTR(X,Y,Z) or SUBSTR(X,Y) or SUBSTRING(X,Y,Z) or SUBSTRING(X,Y) \nDescription: The SUBSTR(X,Y,Z) function returns a substring of input string X that begins with the Y-th character and which is Z characters long. If Z is omitted then SUBSTR(X,Y) returns all characters through the end of the string X beginning with the Y-th. The left-most character of X is number 1. If Y is negative then the first character of the substring is found by counting from the right rather than the left. If Z is negative then the abs(Z) characters preceding the Y-th character are returned. If X is a string then characters indices refer to actual UTF-8 characters. If X is a BLOB then the indices refer to bytes.\"substring()\" is an alias for \"substr()\" beginning with SQLite version 3.34. ",
            "TOTAL_CHANGES() \nDescription: The TOTAL_CHANGES() function returns the number of row changes caused by INSERT, UPDATE or DELETE statements since the current database connection was opened. This function is a wrapper around thesqlite3_total_changes64()C/C++ interface. ",
            "TRIM(X) or TRIM(X,Y) \nDescription: The TRIM(X,Y) function returns a string formed by removing any and all characters that appear in Y from both ends of X. If the Y argument is omitted, TRIM(X) removes spaces from both ends of X. ",
            "TYPEOF(X) \nDescription: The TYPEOF(X) function returns a string that indicates thedatatypeof the expression X: \"null\", \"integer\", \"real\", \"text\", or \"blob\". ",
            "UNHEX(X) or UNHEX(X,Y) \nDescription: The UNHEX(X,Y) function returns a BLOB value which is the decoding of the hexadecimal string X. If X contains any characters that are not hexadecimal digits and which are not in Y, then UNHEX(X,Y) returns NULL. If Y is omitted, it is understood to be an empty string and hence X must be a pure hexadecimal string. All hexadecimal digits in X must occur in pairs, with both digits of each pair beginning immediately adjacent to one another, or else UNHEX(X,Y) returns NULL. If either parameter X or Y is NULL, then UNHEX(X,Y) returns NULL. The X input may contain an arbitrary mix of upper and lower case hexadecimal digits. Hexadecimal digits in Y have no affect on the translation of X. Only characters in Y that are not hexadecimal digits are ignored in X.See also:hex() ",
            "UNICODE(X) \nDescription: The UNICODE(X) function returns the numeric unicode code point corresponding to the first character of the string X. If the argument to UNICODE(X) is not a string then the result is undefined. ",
            "UNLIKELY(X) \nDescription: The UNLIKELY(X) function returns the argument X unchanged. The UNLIKELY(X) function is a no-op that the code generator optimizes away so that it consumes no CPU cycles at run-time (that is, during calls tosqlite3_step()). The purpose of the UNLIKELY(X) function is to provide a hint to the query planner that the argument X is a boolean value that is usually not true. The UNLIKELY(X) function is equivalent tolikelihood(X, 0.0625). ",
            "UPPER(X) \nDescription: The UPPER(X) function returns a copy of input string X in which all lower-case ASCII characters are converted to their upper-case equivalent. ",
            "ZEROBLOB(N) \nDescription: The ZEROBLOB(N) function returns a BLOB consisting of N bytes of 0x00. SQLite manages these zeroblobs very efficiently. Zeroblobs can be used to reserve space for a BLOB that is later written usingincremental BLOB I/O. This SQL function is implemented using thesqlite3_result_zeroblob()routine from the C/C++ interface. ",
            "AVG(X) \nDescription: The AVG() function returns the average value of all non-NULLXwithin a group. String and BLOB values that do not look like numbers are interpreted as 0. The result of AVG() is always a floating point value whenever there is at least one non-NULL input even if all inputs are integers. The result of AVG() is NULL if there are no non-NULL inputs. The result of AVG() is computed astotal()/count()so all of the constraints that apply tototal()also apply to AVG(). ",
            "COUNT(X) or COUNT(*) \nDescription: The COUNT(X) function returns a count of the number of times thatXis not NULL in a group. The COUNT(*) function (with no arguments) returns the total number of rows in the group. ",
            "GROUP_CONCAT(X) or GROUP_CONCAT(X,Y) or STRING_AGG(X,Y) \nDescription: The GROUP_CONCAT() function returns a string which is the concatenation of all non-NULL values ofX. If parameterYis present then it is used as the separator between instances ofX.A comma (\",\") is used as the separator ifYis omitted.The string_agg(X,Y) function is an alias for GROUP_CONCAT(X,Y). String_agg() is compatible with PostgreSQL and SQL-Server and GROUP_CONCAT() is compatible with MySQL.The order of the concatenated elements is arbitrary unless an ORDER BY argument is included immediately after the last parameter. ",
            "MAX(X) \nDescription: The MAX() aggregate function returns the maximum value of all values in the group. The maximum value is the value that would be returned last in an ORDER BY on the same column. Aggregate MAX() returns NULL if and only if there are no non-NULL values in the group. ",
            "MIN(X) \nDescription: The MIN() aggregate function returns the minimum non-NULL value of all values in the group. The minimum value is the first non-NULL value that would appear in an ORDER BY of the column. Aggregate MIN() returns NULL if and only if there are no non-NULL values in the group. ",
            "SUM(X) or TOTAL(X) \nDescription: The SUM() and TOTAL() aggregate functions return the sum of all non-NULL values in the group. If there are no non-NULL input rows then SUM() returns NULL but TOTAL() returns 0.0. NULL is not normally a helpful result for the sum of no rows but the SQL standard requires it and most other SQL database engines implement SUM() that way so SQLite does it in the same way in order to be compatible. The non-standard TOTAL() function is provided as a convenient way to work around this design problem in the SQL language. ",
            "ROW_NUMBER() \nDescription: The number of the row within the current partition. Rows are numbered starting from 1 in the order defined by the ORDER BY clause in the window definition, or in arbitrary order otherwise. ",
            "RANK() \nDescription: The row_number() of the first peer in each group - the rank of the current row with gaps. If there is no ORDER BY clause, then all rows are considered peers and this function always returns 1. ",
            "DENSE_RANK() \nDescription: The number of the current row's peer group within its partition - the rank of the current row without gaps. Rows are numbered starting from 1 in the order defined by the ORDER BY clause in the window definition. If there is no ORDER BY clause, then all rows are considered peers and this function always returns 1. ",
            "PERCENT_RANK() \nDescription: Despite the name, this function always returns a value between 0.0 and 1.0 equal to (rank- 1)/(partition-rows- 1), whererankis the value returned by built-in window function rank() andpartition-rowsis the total number of rows in the partition. If the partition contains only one row, this function returns 0.0. ",
            "CUME_DIST() \nDescription: The cumulative distribution. Calculated asrow-number/partition-rows, whererow-numberis the value returned by row_number() for the last peer in the group andpartition-rowsthe number of rows in the partition. ",
            "NTILE(N) \nDescription: ArgumentNis handled as an integer. This function divides the partition into N groups as evenly as possible and assigns an integer between 1 andNto each group, in the order defined by the ORDER BY clause, or in arbitrary order otherwise. If necessary, larger groups occur first. This function returns the integer value assigned to the group that the current row is a part of. ",
            "LAG(expr) or LAG(expr, offset) or LAG(expr, offset, default) \nDescription: The first form of the LAG() function returns the result of evaluating expressionexpragainst the previous row in the partition. Or, if there is no previous row (because the current row is the first), NULL. ",
            "LEAD(expr) or LEAD(expr, offset) or LEAD(expr, offset, default) \nDescription: The first form of the LEAD() function returns the result of evaluating expressionexpragainst the next row in the partition. Or, if there is no next row (because the current row is the last), NULL. ",
            "FIRST_VALUE(expr) \nDescription: This built-in window function calculates the window frame for each row in the same way as an aggregate window function. It returns the value ofexprevaluated against the first row in the window frame for each row. ",
            "LAST_VALUE(expr) \nDescription: This built-in window function calculates the window frame for each row in the same way as an aggregate window function. It returns the value ofexprevaluated against the last row in the window frame for each row. ",
            "NTH_VALUE(expr, N) \nDescription: This built-in window function calculates the window frame for each row in the same way as an aggregate window function. It returns the value ofexprevaluated against the rowNof the window frame. Rows are numbered within the window frame starting from 1 in the order defined by the ORDER BY clause if one is present, or in arbitrary order otherwise. If there is noNth row in the partition, then NULL is returned. ",
            "ACOS(X) \nDescription: Return the arccosine of X. The result is in radians. ",
            "ACOSH(X) \nDescription: Return the hyperbolic arccosine of X. ",
            "ASIN(X) \nDescription: Return the arcsine of X. The result is in radians. ",
            "ASINH(X) \nDescription: Return the hyperbolic arcsine of X. ",
            "ATAN(X) \nDescription: Return the arctangent of X. The result is in radians. ",
            "ATAN2(Y,X) \nDescription: Return the arctangent of Y/X. The result is in radians. The result is placed into correct quadrant depending on the signs of X and Y. ",
            "ATANH(X) \nDescription: Return the hyperbolic arctangent of X. ",
            "CEIL(X) or CEILING(X) \nDescription: Return the first representable integer value greater than or equal to X. For positive values of X, this routine rounds away from zero. For negative values of X, this routine rounds toward zero. ",
            "COS(X) \nDescription: Return the cosine of X. X is in radians. ",
            "COSH(X) \nDescription: Return the hyperbolic cosine of X. ",
            "DEGREES(X) \nDescription: Convert value X from radians into degrees. ",
            "EXP(X) \nDescription: Computee(Euler's number, approximately 2.71828182845905) raised to the power X. ",
            "FLOOR(X) \nDescription: Return the first representable integer value less than or equal to X. For positive numbers, this function rounds toward zero. For negative numbers, this function rounds away from zero. ",
            "LN(X) \nDescription: Return the natural logarithm of X. ",
            "LOG(X) or LOG10(X) or LOG(B,X) \nDescription: Return the base-10 logarithm for X. Or, for the two-argument version, return the base-B logarithm of X.Compatibility note: SQLite works like PostgreSQL in that the LOG() function computes a base-10 logarithm. Most other SQL database engines compute a natural logarithm for LOG(). In the two-argument version of LOG(B,X), the first argument is the base and the second argument is the operand. This is the same as in PostgreSQL and MySQL, but is reversed from SQL Server which uses the second argument as the base and the first argument as the operand. ",
            "LOG2(X) \nDescription: Return the logarithm base-2 for the number X. ",
            "MOD(X,Y) \nDescription: Return the remainder after dividing X by Y. This is similar to the '%' operator, except that it works for non-integer arguments. ",
            "PI() \nDescription: Return an approximation for π. ",
            "POW(X,Y) or POWER(X,Y) \nDescription: Compute X raised to the power Y. ",
            "RADIANS(X) \nDescription: Convert X from degrees into radians. ",
            "SIN(X) \nDescription: Return the sine of X. X is in radians. ",
            "SINH(X) \nDescription: Return the hyperbolic sine of X. ",
            "SQRT(X) \nDescription: Return the square root of X. NULL is returned if X is negative. ",
            "TAN(X) \nDescription: Return the tangent of X. X is in radians. ",
            "TANH(X) \nDescription: Return the hyperbolic tangent of X. ",
            "TRUNC(X) \nDescription: Return the representable integer in between X and 0 (inclusive) that is furthest away from zero. Or, in other words, return the integer part of X, rounding toward zero. The TRUNC() function is similar toceiling(X)andfloor(X)except that it always rounds toward zero whereas ceiling(X) and floor(X) round up and down, respectively. ",
            "DATE(time-value, modifier, modifier, ...) \nDescription: Returns the date as text in this format: YYYY-MM-DD. ",
            "TIME(time-value, modifier, modifier, ...) \nDescription: Returns the time as text in formatted as HH:MM:SS or as HH:MM:SS.SSS if the subsec modifier is used. ",
            "DATETIME(time-value, modifier, modifier, ...) \nDescription: Returns the date and time formatted as YYYY-MM-DD HH:MM:SS or as YYYY-MM-DD HH:MM:SS.SSS if the subsec modifier is used. ",
            "JULIANDAY(time-value, modifier, modifier, ...) \nDescription: Returns the Julian day - the fractional number of days since noon in Greenwich on November 24, 4714 B.C. (Proleptic Gregorian calendar). ",
            "UNIXEPOCH(time-value, modifier, modifier, ...) \nDescription: Returns a unix timestamp - the number of seconds since 1970-01-01 00:00:00 UTC. The UNIXEPOCH() function normally returns an integer number of seconds, but with the optional subsec modifier it will return a floating point number which is the fractional number of seconds. ",
            "STRFTIME(format, time-value, modifier, modifier, ...) \nDescription: Returns the date formatted according to the format string specified as the first argument. The format string supports the most common substitutions found in the STRFTIME() function from the standard C library plus two new substitutions, %f and %J. ",
            "TIMEDIFF(time-value, time-value) \nDescription: Returns a string that describes the amount of time that must be added to B in order to reach time A. The format of the TIMEDIFF() result is designed to be human-readable. "
        ]
        return funcs


class QuestionGenerationPrompt:
    def __init__(self):
        pass

    def question_synthesis_prompt(self, style_desc, engine, column_info, sql, steps, guidelines, output_format, instruction):
        template = '''**Task Overview**
        Your task is to create a high-quality natural language question based on a given SQL query and other information.

        **Style**
        The natural language question should follow this style:
        {style_desc}

        **Database Engine**
        {engine}

        **Column Information**
        Below are column names and their corresponding descriptions:
        {column_info}

        **SQL Query**
        Given SQL query:
        ```sql
        {sql}
        ```

        **Reasoning Steps**
        {steps}

        **Guidelines**
        {guidelines}

        **Output Format**
        {output_format}

        **Insturction**
        {instruction}
        '''
        return template.format(
            style_desc = style_desc,
            engine = engine,
            column_info = column_info,
            sql = sql,
            steps = steps,
            guidelines = guidelines,
            output_format = output_format,
            instruction = instruction
        )  

    def get_style2desc(self):
        template = {
        "Formal": '''**Formal Style**
        - Uses standard grammar and vocabulary.
        - Example: Find all students older than 18 years and return their home addresses.''',

        "Colloquial": '''**Colloquial Style**
        - Employs informal vocabulary and expressions.
        - Example: Hey! Could you help me find all the students who are over 18? I'd love to know their names and where they live.''',

        "Imperative": '''**Imperative Style**
        - Uses command or directive sentences.
        - Example: Could you please gather all the students who are older than 18? I really need to know their names and where they live!''',

        "Interrogative": '''**Interrogative Style**
        - Uses question forms.
        - Example: Could you tell me which students are older than 18 and what their home addresses are?''',

        "Descriptive": '''**Descriptive Style**
        - Uses detailed descriptions with contextual information.
        - Example: I want to know the names and home addresses of all students older than 18.''',

        "Concise": '''**Concise Style**
        - Use short sentences.
        - Example: Students older than 18, return their names and addresses.''',

        "Vague": '''**Vague Style**
        - Includes ambiguous vocabulary requiring inference.
        - Example: What are the names and addresses of those older students? (External Knowledge: 'older students' refers to age >= 18.)''',

        "Metaphorical": '''**Metaphorical Style**
        - Uses metaphors or metaphorical expressions.
        - Example: Find the names and addresses of those who have reached adulthood. (External Knowledge: 'reached adulthood' refers to age >= 18.)'''
        }
        return template

    def get_steps_wo_ek(self):
        template = '''1. **Explain the SQL Query:** Provide a detailed explanation of what the query does.
        2. **Generate a Question:** Formulate a natural language question based on the SQL query and explanation.'''
        return template

    def get_steps_w_ek(self):
        template = '''1. **Explain the SQL Query:** Provide a detailed explanation of what the query does.
        2. **Generate a Question:** Formulate a natural language question based on the SQL query and explanation.
        3. **External Knowledge:** For Vague or Metaphorical styles, include external knowledge to enhance clarity.'''
        return template

    def get_steps_multi_round(self):
        template = '''1. **Explain the SQL Query:** Provide a detailed explanation of what the query does.
        2. **Generate a Dialogue:** Create a conversation between the User and the Assistant based on the SQL query and its explanation.'''
        return template

    def get_guidelines_wo_ek(self):
        template = '''1. Clearly describe the columns being selected by the SQL query. For example:
        - "SELECT * ... FROM ..." means "Find all ...";
        - "SELECT f.check_date, f.status, f.remarks, c.year, c.year_min, c.year_max, c.year_average, c.data_quality_score FROM ..." means "Return the check dates, statuses, remarks, years, minimum years, maximum years, average years, and quality scores for ...".
        2. Ensure the natural language question accurately captures the semantics of the SQL query, including conditions such as predicates, `ORDER BY`, and `LIMIT` clauses.'''
        return template

    def get_guidelines_w_ek(self):
        template = '''1. Clearly describe the columns being selected by the SQL query. For example:
        - "SELECT * ... FROM ..." means "Find all ...";
        - "SELECT f.check_date, f.status, f.remarks, c.year, c.year_min, c.year_max, c.year_average, c.data_quality_score FROM ..." means "Return the check dates, statuses, remarks, years, minimum years, maximum years, average years, and quality scores for ...".
        2. Ensure the natural language question accurately captures the semantics of the SQL query, including conditions such as predicates, `ORDER BY`, and `LIMIT` clauses.
        3. If necessary, incorporate external knowledge using multiple entries separated by semicolons (";"). These can include formulas, common sense, domain-specific knowledge, or extended context, such as information from long documents. Each entry should be concise.'''
        return template

    def get_guidelines_multi_round(self):
        template = '''1. Clearly describe the columns being selected by the SQL query. For example:
        - "SELECT * ... FROM ..." means "Find all ...";
        - "SELECT f.check_date, f.status, f.remarks, c.year, c.year_min, c.year_max, c.year_average, c.data_quality_score FROM ..." means "Return the check dates, statuses, remarks, years, minimum years, maximum years, average years, and quality scores for ...".
        2. Ensure the conversation accurately captures the semantics of the SQL query, including conditions such as predicates, `ORDER BY`, and `LIMIT` clauses.'''
        return template

    def get_output_format_wo_ek(self):
        template = '''Please structure your response as follows:

        [EXPLANATION-START]
        (SQL Explanation)
        [EXPLANATION-END]

        [QUESTION-START]
        (Natural Language Question)
        [QUESTION-END]

        - **SQL Explanation**: Provide a clear and detailed explanation of the SQL query, enclosed within [EXPLANATION-START] and [EXPLANATION-END].
        - **Natural Language Question**: Translate the SQL query into a natural language question, enclosed within [QUESTION-START] and [QUESTION-END].'''
        return template

    def get_output_format_w_ek(self):
        template = '''Please structure your response as follows:

        [EXPLANATION-START]
        (SQL Explanation)
        [EXPLANATION-END]

        [QUESTION-START]
        (Natural Language Question)
        [QUESTION-END]

        [EXTERNAL-KNOWLEDGE-START]
        (External Knowledge)
        [EXTERNAL-KNOWLEDGE-END]

        - **SQL Explanation**: Provide a clear and detailed explanation of the SQL query, enclosed within [EXPLANATION-START] and [EXPLANATION-END].
        - **Natural Language Question**: Translate the SQL query into a natural language question, enclosed within [QUESTION-START] and [QUESTION-END].
        - **External Knowledge**: Include any relevant external knowledge if applicable, enclosed within [EXTERNAL-KNOWLEDGE-START] and [EXTERNAL-KNOWLEDGE-END]. Leave this section blank if not needed.'''
        return template

    def get_output_format_multi_round(self):
        template = '''Please structure your response as follows:

        [EXPLANATION-START]
        (SQL Explanation)
        [EXPLANATION-END]

        [QUESTION-START]
        (Natural Language Question, in the format of [{"User": ...}, {"Assistant": ...}, {"User": ...}, ....])
        [QUESTION-END]

        - **SQL Explanation**: Provide a clear and detailed explanation of the SQL query, enclosed within [EXPLANATION-START] and [EXPLANATION-END].
        - **Natural Language Question**: Convert the SQL query into a multi-round dialogue, enclosed within [QUESTION-START] and [QUESTION-END]. Represent this as a list that captures multiple rounds of conversation between the User and the Assistant.'''
        return template

    def get_instruction_wo_ek(self):
        template = "Based on the above information, follow the reasoning steps to generate the explanation and the question corresponding to the SQL query."
        return template

    def get_instruction_w_ek(self):
        template = "Based on the above information, follow the reasoning steps to generate the explanation, the question, and the external knowledge corresponding to the SQL query."
        return template

    def get_instruction_multi_round(self):
        template = "Based on the above information, follow the reasoning steps to generate the explanation and the dialogue corresponding to the SQL query."
        return template

class SQLVariationPrompt:
    def __init__(self):
        pass

    def variation_type_prompt(self, variation_type: int):
        type_prompts = [
            '''
            Data Value Transformations
            - Alter filter conditions, date ranges, or numerical thresholds
            - Change sorting criteria or limit values
            - Modify aggregation boundaries (e.g., GROUP BY different time periods)
            ''',

            '''Query Structure Modifications
            - Convert aggregation queries to window functions or vice versa
            - Change from simple queries to subqueries or CTEs
            - Transform JOINs to EXISTS/IN clauses or vice versa
            - Switch between correlated and non-correlated subqueries
            ''',

            '''Business Logic Changes
            - Adapt the query for different business scenarios (sales → inventory, customers → suppliers)
            - Modify to handle different data granularities (daily → monthly, individual → grouped)
            - Change the analytical perspective (profit analysis → cost analysis)
            - Alter the metrics being calculated (sum → average, count → percentage)
            ''',

            '''Complexity Enhancements
            - Add extra filtering conditions or business rules
            - Introduce additional table joins
            - Include conditional logic with CASE statements
            - Add data validation or quality checks
            ''',

            '''Advanced SQL Features
            - Implement complex window functions with partitioning
            - Create queries requiring UNION/INTERSECT/EXCEPT operations
            - Add recursive CTEs for hierarchical data
            - Include pivot/unpivot operations
            ''',

            '''Performance and Optimization
            - Add performance optimization hints
            - Restructure for better index usage
            - Convert to more efficient query patterns
            - Add appropriate WHERE clause optimizations
            ''',
        ]
        return type_prompts[variation_type]

    def insert_stmts_template(self, insert_statements):
        template = '''### INSERT INTO Statements
        Below are several `INSERT INTO` statements. Use these to help generate predicates (i.e., `WHERE` clauses) in your SQL query:
        {insert_statements}
        '''
        return template.format(insert_statements=insert_statements)

    def sql_variation_prompt(self, original_sql, schema_str, db_value_prompt, variation_prompt, db_engine):
        template = """**Task Overview**
        Create a new reasonable and executable SQL query by applying the specified transformations to the original query.

        **Database Engine**
        {db_engine}

        **Database Schema**
        {schema_str}

        {db_value_prompt}

        **Original SQL Query**
        ```sql
        {original_sql}
        ```

        **Transformation Instructions**
        {variation_prompt}

        **Requirements**
        1. The new query must be syntactically correct for {db_engine}
        2. All referenced tables/columns must exist in the provided schema
        3. Ensure the query is executable

        **Output Format**
        The transformed SQL query should be enclosed in a code block:
        ```sql
        -- Your transformed SQL query here
        ```

        **Answer**
        Let's proceed step by step.
        """
        return template.format(
            variation_prompt=variation_prompt,
            schema_str=schema_str,
            db_value_prompt=db_value_prompt,
            original_sql=original_sql,
            db_engine=db_engine
        )