import json

from openai import AsyncOpenAI
from .MCPClient import MCPClient

import os


client = AsyncOpenAI(base_url="http://localhost:1234/v1", api_key=os.getenv("API_KEY"))
mcp_client = MCPClient()
model = "meta-llama-3.1-8b-instruct"

# chat function
async def chat(user_input):
    """
    Processes user input through a two-step LLM interaction with tool integration.

    This function performs the following steps:
    1. Connects to Gmail MCP server and retrieves available tools
    2. Makes initial LLM call to determine which tool to use
    3. Executes the selected tool with provided arguments
    4. Makes second LLM call to generate final response based on tool output

    Args:
        user_input (str): The input message from the user to be processed

    Returns:
        str: The final response message from the LLM

    Raises:
        None
    """

    await mcp_client.connect_to_server("http://localhost:8931/sse")
    response = await mcp_client.get_tools()    

    tools = [{
    "type": "function",
    "function": {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.inputSchema
    }
    } for tool in response.tools]

    # 1st LLM call to determine which tool to use
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Always answer in Vietnamese."},
            {"role": "user", "content": user_input}
        ],
        tools=tools
    )

    # if LLM decides to use a tool
    if response.choices[0].message.tool_calls:        
        tool_name = response.choices[0].message.tool_calls[0].function.name
        tool_args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        print(f"Tool Used: {tool_name}, Arguments: {tool_args}")

        # execute the tool called by the LLM
        tool_response = await mcp_client.session.call_tool(tool_name, tool_args)
        tool_response_text = tool_response.content[0].text    

        # 2nd LLM call to determine final response
        res = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Always answer in Vietnamese."},
                {"role": "user", "content": user_input},
                {"role": "tool", "name": tool_name, "content": tool_response_text},
            ]        
        )

        response = res.choices[0].message.content
        
    # if LLM decides not to use a tool
    else:
        response = response.choices[0].message.content

    # disconnect from the server
    await mcp_client.disconnect()
    
    return response   