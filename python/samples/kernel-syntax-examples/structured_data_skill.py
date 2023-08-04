# Copyright (c) Microsoft. All rights reserved.


import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.core_skills import DataSkill
import asyncio
import pandas as pd

async def main() -> None:
    kernel = sk.Kernel()
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_text_completion_service(
        "davinci-003", sk_oai.OpenAITextCompletion("text-davinci-003", api_key, org_id)
    )
    data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 32, 28, 22, 29],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami'],
    'Salary': [60000, 75000, 52000, 48000, 67000]
     }
    df = pd.DataFrame(data)
    data_skill = kernel.import_skill(DataSkill(df), skill_name="data")
    get_pandas_df = data_skill["getPandasDF"]
    json = await get_pandas_df.invoke_async(df)
    context = sk.ContextVariables()
    context["data"] = json
    prompt = "How old is Bob and where does Eve live?"
    qna = kernel.create_semantic_function(prompt, temperature=0.2)
    result = await qna.invoke_async(context=context)
    print(result)
    prompt = "What is the average age?"
    qna = kernel.create_semantic_function(prompt, temperature=0.2)
    result = await qna.invoke_async(context=context)

if __name__ == "__main__":
    asyncio.run(main())