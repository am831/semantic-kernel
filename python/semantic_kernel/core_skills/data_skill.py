# Copyright (c) Microsoft. All rights reserved.

import json

import pandas as pd


class DataSkill:
    """
    Description: A skill that converts structured data to a format an LLM can
    read. Allows the user to perform natural language queries on structured
    data.

    Usage:
        kernel.import_skill(DataSkill(), skill_name="DataSkill")
    """

    PROMPT = """You are designed for data science and interacting with JSON.
    Your goal is to answer queries about data by interacting with the JSON"""

    @staticmethod
    async def csv_to_json(path: str) -> str:
        """
        Returns the csv file as a json object.

        :param path: path to the csv file
        :return: json object
        """

        data = pd.read_csv(path).to_json(orient="records")
        return DataSkill.PROMPT + "\n" + json.dumps(data)

    @staticmethod
    async def pandas_to_json(data: pd.DataFrame) -> str:
        """
        Returns the pandas dataframe as a json object.

        :param data: pandas dataframe
        :return: json object
        """

        data = data.to_json(orient="records")
        jsonstr = json.dumps(data)
        jsonstr = jsonstr[1:-1]
        return DataSkill.PROMPT + jsonstr
