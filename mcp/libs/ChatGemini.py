from google import genai
from google.genai import types

from .MCPClient import MCPClient

import os

#Get API key at https://aistudio.google.com/app/apikey
client = genai.Client(api_key=os.getenv("API_KEY"))
mcp_client = MCPClient()
model = "gemini-2.0-flash"

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

    tools = [
        types.Tool(
            function_declarations=[
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        k: v
                        for k, v in tool.inputSchema.items()
                        if k not in ["additionalProperties", "$schema"]
                    },
                }
            ]
        )
        for tool in response.tools
    ]

    contents = [
        types.Content(
            role="user", parts=[types.Part(text=user_input)]
        )
    ]

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0,
            tools=tools,
        ),
    )

    # Check for a function call
    if response.candidates[0].content.parts[0].function_call:
        tool_call = response.candidates[0].content.parts[0].function_call
        # print(tool_call)

        # Call the MCP server with the predicted tool
        result = await mcp_client.session.call_tool(
            tool_call.name, arguments=tool_call.args
        )

        print(">>>>DEBUG: tool_call result:")
        print(result.content[0].text)

        # Create a function response part
        function_response_part = types.Part.from_function_response(
            name=tool_call.name,
            response={"result": result},
        )

        # Append function call and result of the function execution to contents
        contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)])) # Append the model's function call message
        contents.append(types.Content(role="user", parts=[function_response_part])) # Append the function response

        final_response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
            temperature=0,
            tools=tools,
            ),contents=contents,
        )

        print(final_response.text)
    else:
        print("No function call found in the response.")
        print(response.text)

    # disconnect from the server
    await mcp_client.disconnect()
    
    return response   