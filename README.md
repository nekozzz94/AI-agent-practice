# AI-agent-practice
## 1. MCP demo:
### 1.1 Start MCP playwright server:

```bash
#install npx
sudo apt install npm
mkdir npx
sudo npx playwright install-deps
npx playwright install chrome

#start MCP server
#https://github.com/microsoft/playwright-mcp
npx @playwright/mcp@latest --port 8931
```

![alt text](mcp.png)

### 1.2 Start LLM server with LLM Studio
### 1.3 Run AI agent

![alt text](chat.png)
