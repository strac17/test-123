import os
import glob
from typing import Any

import requests

API_URL = "https://ollama.com/api/chat"
HEADERS = {"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"]}

read_file_tool_schema = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Use this when the user wants read a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "the path of the file to read"},
            },
            "required": ["path"]
        }
    }
}

list_files_schema = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "Use this when the user wants to list files in the current working directory by default, or in a named directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "the path of the directory on which to list the files."}
            },
            "required": []
        }
    }
}

write_file_tool = {
    'type': 'function',
    'function': {
        'name': 'write_file',
        'description': 'Write a new file.  If the file at path does not exist, it will be created as new_file.',
        'parameters': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'The path of the file to create'
                },
                'new_file': {
                    'type': 'string',
                    'description': 'The new text.'
                }
            },
            'required': ['path', 'new_file']
        }
    }
}

def agent(messages):
    tools = [
        read_file_tool_schema, list_files_schema
    ]

    payload = {
        "model": "gpt-oss:120b",
        "messages": messages,
        "stream": False,
        "tools": tools
    }
    response = requests.post(
        API_URL,
        headers=HEADERS,
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    message = data["message"]
    return message

def run_agent(messages):
    tool_called = True
    while tool_called:
        message = agent(messages)
        tool_called = "tool_calls" in message

        if tool_called:
            print("tool called!")
            tool_calls = message["tool_calls"]
            tool_call = tool_calls[0]

            message["tool_calls"] = [tool_call]   # trim to the one call you'll answer
            messages.append(message)              # record the assistant turn (replaces the earlier append)

            function_name = tool_call["function"]["name"]
            function_arguments = tool_call["function"]["arguments"]
            print("tool called:", function_name)

            output = ""

            if function_name == "list_files":
                path = "."
                if "path" in function_arguments:
                    path = function_arguments["path"]
                print(f"arguments for list_files: {path}")
                files = "\n".join(os.listdir(path))
                #files = glob.glob(f"{path}/*")
                output = files

            if function_name == "read_file":
                path = function_arguments["path"]
                with open(path,"r") as file:
                    file_content = file.read()
                output = file_content
            tool_message = {
                "role": "tool",
                "content": f"The output of tool {function_name} is: {output}"
            }
            messages.append(tool_message)
            
        else:
            ai_message = {
                "role": message["role"],
                "content": message["content"]
            }
            messages.append(ai_message)  # inside the else

messages = [
    {"role": "user", 
     "content": """
        You are a legal research assistant.

        First, use list_files to identify the files in the
        breachresponse folder.

        Then use read_file to read each .txt file.
        Do not attempt to read directories or Python files.

        Once you have read the files, provide a concise
        summary of each statute and its relevant provisions.

        IMPORTANT INSTRUCTIONS FOR FUTURE QUESTIONS:

        When I provide a fact pattern, analyze it using the
        statutory provisions contained in the files you read.

        For each fact pattern:
        1. Identify which statute and specific provision applies.
        2. Explain why the provision applies to the facts.
        3. Identify the legal obligations or requirements
           imposed by that provision.
        4. Explain what response or action is required.
        5. Identify any deadlines, reporting requirements,
           notification obligations, or exceptions.
        6. Cite the filename and specific statutory section
           supporting your answer.

        Do not invent statutory provisions or requirements.
        If the files do not contain enough information to
        answer, explain what information is missing.

        Retain the statutory information for use in all
        subsequent questions.
        """}
]
run_agent(messages)

while True:
    question = input("\nYou (or 'quit'): ")
    if question.strip().lower() in ("quit", "exit"):
        break
    messages.append({"role": "user", "content": question})
    run_agent(messages)

    last_message = messages[-1]
    import pprint
    printer = pprint.pprint
    printer(last_message)