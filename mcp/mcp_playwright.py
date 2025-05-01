import json

from openai import AsyncOpenAI

from typing import Optional
from contextlib import AsyncExitStack 
from mcp import ClientSession
from mcp.client.sse import sse_client

class MCPPlaywrightClient:
    """A client for interacting with Gmail through the Zapier MCP (Model Context Protocol) server.
    
    This client establishes and manages a connection to an MCP server using Server-Sent Events (SSE),
    allowing for tool discovery and execution of Gmail-related operations.
    
    Attributes:
        session (Optional[ClientSession]): The active client session with the MCP server.
        exit_stack (AsyncExitStack): Context manager for handling async resources.
    """
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
    async def connect_to_server(self, zapier_url):
        """Establishes an async connection to the MCP server using SSE transport.
        
        Args:
            zapier_url (str): The URL endpoint of the Zapier MCP server to connect to.
            
        Returns:
            ClientSession: The established client session object.
            
        Raises:
            ConnectionError: If the connection to the server cannot be established.
        """
        # Connect using SSE transport
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(zapier_url)
        )
        read, write = sse_transport
        
        # Create the client session
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        
        # Initialize the session
        await self.session.initialize()       
                
        return self.session
    
    async def get_tools(self):
        """Retrieves and formats available tools from the MCP server.
        
        Fetches the list of available tools from the connected MCP server and converts
        them into OpenAI-compatible function schemas.
        
        Returns:
            list[dict]: A list of tool definitions in OpenAI function calling format.
            Each tool is represented as a dictionary containing:
                - type: The type of the tool (always "function")
                - function: Dictionary containing name, description, and parameters schema
                
        Raises:
            RuntimeError: If called before establishing a server connection.
        """
        response = await self.session.list_tools()
        tool_names = [tool.name for tool in response.tools]
        print(f'Available Server Tools: {tool_names}')
        
        openai_tools_schema = [{
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema
        }
        } for tool in response.tools]
        
        return openai_tools_schema
    
    async def disconnect(self):
        """Cleanly disconnects from the MCP server.
        
        Closes the async exit stack and cleans up the client session.
        After disconnection, the client will need to reconnect before making
        further server requests.
        """
        await self.exit_stack.aclose()
        self.session = None


client = AsyncOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
mcp_client = MCPPlaywrightClient()

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
    tools = await mcp_client.get_tools()    

    model = "meta-llama-3.1-8b-instruct"
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