import streamlit as st
import boto3
import json
import uuid
import pandas as pd
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta

REGION = "us-east-1"

# ---------- AWS CLIENTS ----------
bedrock_client = boto3.client("bedrock-agentcore", region_name=REGION)
ce_client = boto3.client("ce")

# ---------- GET RUNTIME ARN ----------
def get_runtime_arn():
    arn_file = Path(".agent_arn")
    if not arn_file.exists():
        st.error(".agent_arn file not found")
        st.stop()

    with open(arn_file) as f:
        return f.read().strip()

runtime_arn = get_runtime_arn()

# ---------- COST EXPLORER DATA ----------
def get_service_costs():

    end = datetime.today()
    start = end - timedelta(days=30)

    response = ce_client.get_cost_and_usage(
        TimePeriod={
            "Start": start.strftime("%Y-%m-%d"),
            "End": end.strftime("%Y-%m-%d")
        },
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )

    services = []
    costs = []

    for group in response["ResultsByTime"][0]["Groups"]:
        services.append(group["Keys"][0])
        costs.append(float(group["Metrics"]["UnblendedCost"]["Amount"]))

    df = pd.DataFrame({
        "Service": services,
        "Cost": costs
    })

    return df.sort_values("Cost", ascending=False).head(10)

# ---------- MONTHLY TREND ----------
def get_monthly_trend():

    end = datetime.today()
    start = end - timedelta(days=180)

    response = ce_client.get_cost_and_usage(
        TimePeriod={
            "Start": start.strftime("%Y-%m-%d"),
            "End": end.strftime("%Y-%m-%d")
        },
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"]
    )

    months = []
    costs = []

    for result in response["ResultsByTime"]:
        months.append(result["TimePeriod"]["Start"])
        costs.append(float(result["Total"]["UnblendedCost"]["Amount"]))

    return pd.DataFrame({
        "Month": months,
        "Cost": costs
    })

# ---------- AGENT CALL ----------
def invoke_agent(query, session_id):

    response = bedrock_client.invoke_agent_runtime(
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
                pass

    return full_response


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI FinOps Cost Optimization Agent",
    layout="wide"
)

st.title("💰 AI FinOps Cost Optimization Agent")

# ---------- SESSION ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- DASHBOARD ----------
col1, col2 = st.columns(2)

with col1:

    st.subheader("Top AWS Services by Cost")

    df = get_service_costs()

    fig = px.pie(
        df,
        names="Service",
        values="Cost",
        title="Top Service Spend"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    st.subheader("Monthly Cost Trend")

    trend = get_monthly_trend()

    fig2 = px.line(
        trend,
        x="Month",
        y="Cost",
        markers=True,
        title="AWS Cost Trend"
    )

    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------- CHAT ----------
st.subheader("💬 Ask the AI Cost Optimization Agent")

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about your AWS costs...")

if prompt:

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("assistant"):

        response = invoke_agent(
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

    st.code(runtime_arn)

    st.write("Session:")
    st.code(st.session_state.session_id)

    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()

    st.markdown("### Example Prompts")

    st.markdown("""
    • Show top 3 expensive AWS services  
    • Is my AWS cost increasing?  
    • Forecast next month AWS bill  
    • Show S3 spending  
    • Recommend cost optimizations
    """)