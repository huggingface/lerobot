# MCP Robot Server

## Overview

This MCP (Model Context Protocol) server, implemented using `FastMCP` and defined in [`server.py`](mcp/server.py:1), provides a set of resources and tools to interact with simulated or real robotic systems. It allows external applications, such as AI assistants or control scripts, to query robot states and command robot actions via a web API.

## Features

The server exposes functionalities through resources (data endpoints) and tools (action endpoints), defined using `@mcp.resource()` and `@mcp.tool()` decorators in [`server.py`](mcp/server.py).

*   **Configurable Number of Robots**: The server supports a configurable number of robots, managed by the global `NUM_ROBOTS` variable in [`server.py`](mcp/server.py:7).
*   **Resources**:
    *   `GET /robot/{robot_index}/wrist_image`: Retrieves a base64 encoded dummy image from a specific robot's wrist camera. The `robot_index` is a path parameter.
        *   _Note: Currently, these are dummy images, and the response is a base64 encoded string._
    *   `GET /workspace/image`: Retrieves a base64 encoded dummy image from a general workspace camera.
        *   _Note: Currently, this is a dummy image, and the response is a base64 encoded string._
*   **Tools (typically invoked via `POST /tools/{tool_name}` by clients)**:
    *   `pick_place(source: str, target: str) -> str`: Instructs the robot to pick an object from a source location and place it at a target location.
    *   `push(target: str) -> str`: Instructs the robot to push an object at a target location.
    *   `lift(target: str) -> str`: Instructs the robot to lift an object at a target location.
    *   `act(instruction: str) -> str`: Allows sending a generic instruction to the robot.
    *   _Note: These tools currently simulate actions and return dummy success messages as strings._

## Setup and Configuration

```bash
export OPENAI_API_KEY="..."
```

### Running the Server

To run the MCP server, navigate to the repository's root directory and execute the following command:

```bash
pip install uv
cd mcp
uv run mcp install server.py --with numpy --with pillow --with openai
```
