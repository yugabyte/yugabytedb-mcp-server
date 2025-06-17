[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/yugabyte-yugabytedb-mcp-server-badge.png)](https://mseep.ai/app/yugabyte-yugabytedb-mcp-server)

# YugabyteDB MCP Server

An [MCP](https://modelcontextprotocol.io/) server implementation for YugabyteDB that allows LLMs to directly interact with your database.

## Features

- List all tables in the database, including schema and row counts
- Run read-only SQL queries and return results as JSON
- Designed for use with [FastMCP](https://github.com/jlowin/fastmcp) and compatible with MCP clients like Claude Desktop, Cursor, and Windsurf Editor

## Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) installed to manage and run the server
- A running YugabyteDB database
- An [MCP client](https://modelcontextprotocol.io/clients) such as [Claude Desktop](https://claude.ai/download) or [Cursor](https://cursor.sh/)

## Installation

Clone this repository and install dependencies:

```bash
git clone git@github.com:yugabyte/yugabytedb-mcp-server.git
cd yugabytedb-mcp-server
uv sync
```

## Configuration

The server is configured using the following environment variable:

- `YUGABYTEDB_URL`: The connection string for your YugabyteDB database (e.g., `dbname=database_name host=hostname port=5433 user=username password=password`)

Example `.env` file:

```
YUGABYTEDB_URL=postgresql://user:password@localhost:5433/yugabyte
```

## Usage

### Running the Server

You can run the server using uv:

```bash
uv run server.py
```

### MCP Client Configuration

To use this server with an MCP client (e.g., Claude Desktop, Cursor), add it to your MCP client configuration. Example for Cursor:

```json
{
  "mcpServers": {
    "yugabytedb-mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/cloned/yugabytedb-mcp-server/",
        "run",
        "src/server.py"
      ],
      "env": {
        "YUGABYTEDB_URL": "dbname=database_name host=hostname port=5433 user=username password=password load_balance=true topology_keys=cloud.region.zone1,cloud.region.zone2"
      }
    }
  }
}
```

- Replace `/path/to/cloned/yugabytedb-mcp-server/` with the path to your cloned repository.
- Set the correct database URL in the `env` section.

### Claude Desktop

1. Edit the configuration file. Go to Claude -> Settings -> Developer -> Edit Config
2. Add the above configuration under `mcpServers`.
3. Restart Claude Desktop.

#### Claude Desktop Logs

The logs for Claude Desktop can be found in the following locations:

- MacOS: ~/Library/Logs/Claude
- Windows: %APPDATA%\Claude\Logs

The logs can be used to diagnose connection issues or other problems with your MCP server configuration. For more details, refer to the [official documentation](https://modelcontextprotocol.io/quickstart/user#getting-logs-from-claude-for-desktop).

### Cursor

1. Install [Cursor](https://cursor.sh/) on your machine.
2. Go to Cursor > Settings > Cursor Settings > MCP > Add a new global MCP server.
3. Add the configuration as above.
4. Save the configuration.
5. You will see yugabytedb-mcp-server as an added server in MCP servers list. Refresh to see if server is enabled.

#### Cursor Logs

In the bottom panel of Cursor, click on "Output" and select "Cursor MCP" from the dropdown menu to view server logs. This can help diagnose connection issues or other problems with your MCP server configuration.

### Windsurf Editor

1. Install [Windsurf Editor](https://windsurf.com/download) on your machine.
2. Go to Windsurf > Settings > Windsurf Settings > Cascade > Model Context Protocol (MCP) Servers > Add server > Add custom server.
3. Add the configuration as above.
4. Save and refresh.

### Tools Provided

- **summarize_database**: Lists all tables in the database, including schema and row counts.
- **run_read_only_query**: Runs a read-only SQL query and returns the results as JSON.

### Example Usage

Once connected via an MCP client, you can:
- Ask for a summary of the database tables and schemas
- Run SELECT queries and get results in JSON

## Environment Variables

- `YUGABYTEDB_URL`: (required) The connection string for your YugabyteDB/PostgreSQL database

## Troubleshooting

- Ensure the `YUGABYTEDB_URL` is set and correct
- Verify your database is running and accessible
- Check that your user has the necessary permissions
- Make sure `uv` is installed and available in your PATH. Note: If claude is unable to access uv, giving the error: `spawn uv ENOENT`, try symlinking the uv for global access:
```shell
sudo ln -s "$(which uv)" /usr/local/bin/uv
```
- Review logs in your MCP client for connection or query errors

## Development

- Project dependencies are managed in `pyproject.toml`
- Main server logic is in `src/server.py`
