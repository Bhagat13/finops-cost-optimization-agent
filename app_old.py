import streamlit as st
import boto3
import json
import uuid
from pathlib import Path

# ---------- CONFIG ----------
REGION = "us-east-1"

# ---------- GET RUNTIME ARN ----------
def get_runtime_arn():
    arn_file = Path(".agent_arn")
    if not arn_file.exists():
        st.error(".agent_arn file not found. Deploy the agent first.")
        st.stop()

    with open(arn_file, "r") as f:
        return f.read().strip()


# ---------- STREAM AGENT RESPONSE ----------
def invoke_agent(runtime_arn, query, session_id):

    client = boto3.client(
        "bedrock-agentcore",
        region_name=REGION
    )

    response = client.invoke_agent_runtime(
        agentRuntimeArn=runtime_arn,
        qualifier="DEFAULT",
        runtimeSessionId=session_id,
        payload=json.dumps({"prompt": query}),
    )

    full_response = ""

    placeholder = st.empty()

    for line in response["response"].iter_lines():
        if not line:
            continue

        line_str = line.decode("utf-8")

        if line_str.startswith("data: "):
            payload = line_str[6:]

            try:
                event = json.loads(payload)

                if event.get("type") == "chunk":
                    chunk = event.get("data", "")
                    full_response += chunk
                    placeholder.markdown(full_response)

                elif event.get("type") == "complete":
                    break

            except:
                continue

    return full_response


# ---------- UI ----------
st.set_page_config(
    page_title="AWS Cost Optimization Agent",
    page_icon="💰",
    layout="wide"
)

st.title("💰 AWS Cost Optimization AI Agent")

st.caption("Powered by Amazon Bedrock + AgentCore")

runtime_arn = get_runtime_arn()

# ---------- SESSION STATE ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- DISPLAY CHAT HISTORY ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- USER INPUT ----------
prompt = st.chat_input("Ask about your AWS costs...")

if prompt:

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Agent response
    with st.chat_message("assistant"):

        response = invoke_agent(
            runtime_arn,
            prompt,
            st.session_state.session_id
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

# ---------- SIDEBAR ----------
with st.sidebar:

    st.header("Agent Info")

    st.write("Runtime ARN:")
    st.code(runtime_arn)

    st.write("Session ID:")
    st.code(st.session_state.session_id)

    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()

    st.markdown("### Example Questions")

    st.markdown("""
    • Show my top 3 expensive services  
    • Are my AWS costs increasing?  
    • What is my Amazon Bedrock cost?  
    • Show S3 spending last 30 days  
    • Forecast my next month cost  
    """)