# Copyright (c) Microsoft. All rights reserved.

import json
import pandas as pd
from semantic_kernel.skill_definition import sk_function, sk_function_context_parameter


class DataSkill:
    """
    Description: A skill that converts structured data to a format an LLM can
    read. Allows the user to perform natural language queries on structured 
    data.

    Usage:
        kernel.import_skill(StructuredDataSkill(), skill_name="StructuredData")
    """
    _dataframe: "pd.DataFrame"
    _json: str

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """"
        Initializes from a Pandas DataFrame. See from_csv() and from_json() for 
        more ways to initialize.
        """
        self._dataframe = dataframe
        self._json = self._dataframe.to_json()
    
    @staticmethod
    def from_csv(path: str) -> "DataSkill":
        return DataSkill(
            dataframe=pd.read_csv(path)
        )
    
    @staticmethod
    def from_json(json: str) -> "DataSkill":
        return DataSkill(
            dataframe=pd.read_json(json)
        )
    
    @staticmethod
    async def get_csv(self, path: str) -> str:
        """
        Returns the csv file as a json object.

        :param path: path to the csv file
        :return: json object
        """
        prompt = """You are an agent designed to interact with JSON. 
        Your goal is to return a final answer by interacting with the JSON."""
        data = pd.read_csv(path).to_json(orient='records')
        return prompt + "\n" + json.dumps(data)
    
    @staticmethod
    async def get_pandas_df(self, data: pd.DataFrame) -> str:
        """
        Returns the pandas dataframe as a json object.

        :param data: pandas dataframe
        :return: json object
        """
        prompt = """You are an agent designed to interact with JSON. 
        Your goal is to return a final answer by interacting with the JSON."""
        data = data.to_json(orient='records')
        jsonstr = json.dumps(data)
        jsonstr = jsonstr[1:-1]
        return prompt + jsonstr
    