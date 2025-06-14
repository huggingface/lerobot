import asyncio
import argparse
import sys
import numpy as np
import base64
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage
import openai

# Global variable for the robots dictionary
ROBOTS = {
    "robot_red": "red robot",
    "robot_blue": "blue robot"
}

# Initialize FastMCP
mcp = FastMCP("Robot MCP Server")

# Add a resource to expose the robot configuration
@mcp.resource("robot://config")
async def get_robot_config() -> str:
    """Return the current robot configuration."""
    robot_list = [f"{robot_id}: {description}" for robot_id, description in ROBOTS.items()]
    return f"Robots configured:\n" + "\n".join(robot_list)

# VLA Tool: get_workspace_image
@mcp.tool()
async def get_workspace_description(task: str) -> str:

    # TODO: Replace with image from environment
    workspace_img = PILImage.open("/Users/lucawiehe/Desktop/lerobot/mcp/demo.png")
    
    # Convert PIL image to base64
    import io
    buffer = io.BytesIO()
    workspace_img.save(buffer, format='PNG')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Create OpenAI client
    client = openai.OpenAI(api_key="your_api_key_here")
    
    # Prepare robot information for the prompt
    robot_info = "\n".join([f"- {robot_id}: {description}" for robot_id, description in ROBOTS.items()])
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Analyze this workspace image and provide a structured response in the following format:

                                        Objects in the environment:
                                        - {{name1}}: Short description of state
                                        - {{name2}}: Short description of state

                                        Robots:
                                        - {{robot_id1}}: Short description of its current state
                                            • current pose?
                                            • what is closest object?
                                        - {{robot_id2}}: ...

                                        Task State:
                                        A list of steps describing what steps of the task need to be accomplished. In particular, which 
                                        objects need to be manipulated and how.

                                        Please identify the robots in the image and map them to the robot IDs provided. 
                                        Focus on objects, their positions, robot poses, and task-relevant details for 
                                        robotic manipulation. Use the following context:
                                        - Task: {task}
                                        - Available robots in the system: {robot_info}
                                    """
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        description = response.choices[0].message.content
        return description
        
    except Exception as e:
        return f"Error analyzing workspace image: {str(e)}"

# VLA Tool: pick_place
@mcp.tool()
async def pick_place(obj: str, target: str) -> str:
    """
    Pick an object from a source location and place it at a target location.
    Simulates the action and returns a confirmation message.
    """
    template = f"Pick the {obj} and place it at the {target}."

    # TODO: Replace this with actual VLA call
    await asyncio.sleep(0.1)

    return f"Successfully picked from {obj} and placed at {target}"

# VLA Tool: push
@mcp.tool()
async def push(obj: str, target: str) -> str:
    """
    Push an object at a target location.
    Simulates the action and returns a confirmation message.
    """
    # Simulate action
    template = f"Pick the {obj} and place it at the {target}."

    # TODO: Replace this with actual VLA call
    await asyncio.sleep(0.1)

    return f"Successfully pushed from {obj} and placed to {target}"

# VLA Tool: lift
@mcp.tool()
async def lift(target: str) -> str:
    """
    Lift an object at a target location.
    Simulates the action and returns a confirmation message.
    """
    template = f"Pick the {obj} and place it at the {target}."

    # TODO: Replace this with actual VLA call
    await asyncio.sleep(0.1)

    return f"Successfully picked from {obj} and placed at {target}"

# VLA Tool: act
@mcp.tool()
async def act(instruction: str) -> str:
    """
    Execute a generic instruction.
    Simulates the action and returns a confirmation message.
    """
    template = f"Pick the {obj} and place it at the {target}."

    # TODO: Replace this with actual VLA call
    await asyncio.sleep(0.1)

    return f"Successfully executed: {instruction}"

# Updated tool to get current robot information
@mcp.tool()
async def get_robots() -> str:
    """
    Get the current number of robots and their descriptions.
    """
    robot_info = [f"{robot_id}: {description}" for robot_id, description in ROBOTS.items()]
    return f"Total robots: {len(ROBOTS)}\n" + "\n".join(robot_info)

def main():
    print("Starting MCP server...")

if __name__ == "__main__":
    main()