#!/usr/bin/env python3
"""
Interactive Test Client for Deployed LLM-powered Cost Optimization Agent
Runs against AWS Bedrock AgentCore Runtime with conversational session support.
"""

import boto3
import json
import sys
import uuid
from pathlib import Path


def get_runtime_arn():
    """Get the runtime ARN from .agent_arn file"""
    arn_file = Path(".agent_arn")
    if not arn_file.exists():
        print("❌ .agent_arn file not found. Deploy the agent first.")
        sys.exit(1)

    with open(arn_file, "r") as f:
        return f.read().strip()


def invoke_agent(runtime_arn: str, query: str, session_id: str):
    """Invoke agent with streaming response and session support"""

    client = boto3.client("bedrock-agentcore", region_name="us-east-1")

    try:
        response = client.invoke_agent_runtime(
            agentRuntimeArn=runtime_arn,
            qualifier="DEFAULT",
            runtimeSessionId=session_id,
            payload=json.dumps({"prompt": query}),
        )

        full_response = []

        for line in response["response"].iter_lines():
            if not line:
                continue

            line_str = line.decode("utf-8")

            if line_str.startswith("data: "):
                payload = line_str[6:]

                try:
                    event = json.loads(payload)

                    if event.get("type") == "chunk":
                        chunk_text = event.get("data", "")
                        print(chunk_text, end="", flush=True)
                        full_response.append(chunk_text)

                    elif event.get("type") == "complete":
                        break

                except json.JSONDecodeError:
                    continue

        print("\n")
        return "".join(full_response)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def main():
    """Run interactive conversational session"""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     COST OPTIMIZATION AGENT - LIVE CHAT                     ║
║                                                                              ║
║  Running against AWS Bedrock AgentCore Runtime (Production Deployment)      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
Type 'exit' or 'quit' to end the session.
    """)

    runtime_arn = get_runtime_arn()
    print(f"🏷️  Runtime ARN: {runtime_arn}")

    # Generate persistent session ID for conversation
    session_id = str(uuid.uuid4())
    print(f"🧠 Session ID: {session_id}")
    print("🤖 Agent is ready...\n")

    while True:
        try:
            query = input("You: ").strip()

            if query.lower() in ["exit", "quit"]:
                print("\n👋 Ending session. Goodbye!")
                break

            if not query:
                continue

            print("\nAgent:")
            print("-" * 80)

            invoke_agent(runtime_arn, query, session_id)

            print("=" * 80)

        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break


if __name__ == "__main__":
    main()