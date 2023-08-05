# Copyright (c) Microsoft. All rights reserved.


import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.core_skills import DataSkill, MathSkill
import asyncio
import pandas as pd

async def main() -> None:
    kernel = sk.Kernel()
    api_key, org_id = sk.openai_settings_from_dot_env()
    openai_chat_completion = sk_oai.OpenAIChatCompletion(
        "gpt-3.5-turbo", api_key, org_id
    )
    kernel.add_chat_service("chat_service", openai_chat_completion)
    data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 32, 28, 22, 29],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami'],
    'Salary': [60000, 75000, 52000, 48000, 67000]
     }
    df = pd.DataFrame(data)
  
    data_skill = kernel.import_skill(DataSkill(df), skill_name="data")
    json = await DataSkill.get_pandas_df(data_skill, data=df)
    context = sk.ContextVariables()
    #context["user_input"] = json
    prompt = "How old is Bob and where does Eve live?"
    context["user_input"] = json + prompt
    prompt_config = sk.PromptTemplateConfig.from_completion_parameters(
        max_tokens=2000, temperature=0.7, top_p=0.8
    )
    prompt_template = sk.ChatPromptTemplate(  # Create the chat prompt template
        "{{$user_input}}", kernel.prompt_template_engine, prompt_config
    )
    prompt_template.add_system_message(prompt) # Add the memory as a system message
    function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
    query_func = kernel.register_semantic_function(None, "Data", function_config)
    result = await kernel.run_async(query_func, input_vars=context)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())