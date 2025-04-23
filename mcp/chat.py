from mcp_playwright import chat

import asyncio

async def main():
    user_input = 'top news in https://vnexpress.net/?'

    res = await chat(user_input)
    print(res)

asyncio.run(main())