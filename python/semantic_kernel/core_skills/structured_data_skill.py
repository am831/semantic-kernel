# Copyright (c) Microsoft. All rights reserved.

import json
import pandas as pd
from semantic_kernel.skill_definition import sk_function, sk_function_context_parameter


class StructuredDataSkill:
    """
    Description: A skill that converts structured data to a format an LLM can
    read. Allows the user to perform natural language queries on structured 
    data.

    Usage:
        kernel.import_skill(StructuredDataSkill(), skill_name="StructuredData")


    """

    @sk_function(
            description="Converts a CSV file to a JSON object",
            name="getCSV",
            input_description="Path of the csv file",
    )
    @sk_function_context_parameter(
        name="CSV",
        description="CSV to JSON",
    )
    async def get_csv(self, path: str) -> json:
        """
        Returns the csv file as a json object.

        :param path: path to the csv file
        :return: json object
        """
        data = pd.read_csv(path)
        return data.to_json(orient='records')
    
    @sk_function(
        description="Converts a pandas dataframe to a JSON object",
        name="getPandasDF",
        input_description="Pandas dataframe",
    )
    @sk_function_context_parameter(
        name="PandasDF",
        description="Pandas dataframe to JSON",
    )
    async def get_pandas_df(self, data: pd.DataFrame) -> json:
        """
        Returns the pandas dataframe as a json object.

        :param data: pandas dataframe
        :return: json object
        """
        return data.to_json(orient='records')