import streamlit as st
import boto3
import json
import uuid
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

REGION = "us-east-1"

bedrock = boto3.client("bedrock-agentcore", region_name=REGION)
ce = boto3.client("ce")
ec2 = boto3.client("ec2", region_name=REGION)
cw = boto3.client("cloudwatch", region_name=REGION)

# ------------------ LOAD ARN ------------------
def get_runtime_arn():
    return Path(".agent_arn").read_text().strip()

runtime_arn = get_runtime_arn()
# ------------------Detect Idle EC2 Instances -----
def get_idle_ec2_instances():

    instances = []

    res = ec2.describe_instances()

    for r in res["Reservations"]:
        for inst in r["Instances"]:

            instance_id = inst["InstanceId"]

            # Get CPU Utilization
            metrics = cw.get_metric_statistics(
                Namespace="AWS/EC2",
                MetricName="CPUUtilization",
                Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
                StartTime=datetime.utcnow() - timedelta(days=1),
                EndTime=datetime.utcnow(),
                Period=3600,
                Statistics=["Average"],
            )

            datapoints = metrics["Datapoints"]

            if datapoints:
                avg_cpu = sum(d["Average"] for d in datapoints) / len(datapoints)

                if avg_cpu < 5:  # Idle threshold
                    instances.append({
                        "InstanceId": instance_id,
                        "AvgCPU": round(avg_cpu, 2),
                        "State": inst["State"]["Name"]
                    })

    return pd.DataFrame(instances)
# ------------------ Stop EC2 Instances ---------
def stop_instances(instance_ids):

    ec2.stop_instances(InstanceIds=instance_ids)

    return f"Stopping instances: {', '.join(instance_ids)}"

# ------------------ SESSION ------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------ COST DATA ------------------
def get_service_costs():
    end = datetime.today()
    start = end - timedelta(days=30)

    res = ce.get_cost_and_usage(
        TimePeriod={"Start": start.strftime("%Y-%m-%d"), "End": end.strftime("%Y-%m-%d")},
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )

    data = []
    for g in res["ResultsByTime"][0]["Groups"]:
        data.append({
            "Service": g["Keys"][0],
            "Cost": float(g["Metrics"]["UnblendedCost"]["Amount"])
        })

    return pd.DataFrame(data).sort_values("Cost", ascending=False).head(10)

def get_daily_trend():
    end = datetime.today()
    start = end - timedelta(days=14)

    res = ce.get_cost_and_usage(
        TimePeriod={"Start": start.strftime("%Y-%m-%d"), "End": end.strftime("%Y-%m-%d")},
        Granularity="DAILY",
        Metrics=["UnblendedCost"]
    )

    days, costs = [], []

    for r in res["ResultsByTime"]:
        days.append(r["TimePeriod"]["Start"])
        costs.append(float(r["Total"]["UnblendedCost"]["Amount"]))

    return pd.DataFrame({"Day": days, "Cost": costs})

# ------------------ ANOMALY DETECTION ------------------
def detect_anomaly(df):
    mean = df["Cost"].mean()
    std = df["Cost"].std()

    threshold = mean + 2 * std
    anomalies = df[df["Cost"] > threshold]

    return anomalies

# ------------------ FORECAST ------------------
def forecast_cost(df):
    avg = df["Cost"].mean()
    forecast = avg * 30
    return round(forecast, 2)

# ------------------ RECOMMENDATIONS ------------------
def generate_recommendations(df):

    recs = []

    for _, row in df.head(3).iterrows():
        service = row["Service"]
        cost = row["Cost"]

        recs.append(f"{service}: Optimize usage (~${round(cost*0.2,2)} savings possible)")

    return recs

# ------------------ AGENT CALL ------------------
def invoke_agent(prompt):

    res = bedrock.invoke_agent_runtime(
        agentRuntimeArn=runtime_arn,
        qualifier="DEFAULT",
        runtimeSessionId=st.session_state.session_id,
        payload=json.dumps({"prompt": prompt})
    )

    full = ""
    tools = []
    box = st.empty()

    for line in res["response"].iter_lines():

        if not line:
            continue

        line = line.decode("utf-8")

        if line.startswith("data: "):
            payload = line[6:]

            try:
                event = json.loads(payload)

                if event.get("type") == "tool_call":
                    tools.append(event)

                if event.get("type") == "chunk":
                    chunk = event.get("data", "")
                    full += chunk
                    box.markdown(full)

                if event.get("type") == "complete":
                    break

            except:
                pass

    return full, tools

# ------------------ UI ------------------

st.set_page_config(layout="wide")
st.title("💰 GenAI FinOps Copilot")

# ---------- DASHBOARD ----------
col1, col2 = st.columns(2)

service_df = get_service_costs()
trend_df = get_daily_trend()

with col1:
    st.subheader("Top Services")
    st.plotly_chart(px.pie(service_df, names="Service", values="Cost"))

with col2:
    st.subheader("Daily Cost Trend")
    st.plotly_chart(px.line(trend_df, x="Day", y="Cost", markers=True))
st.subheader("🛑 Idle EC2 Detection & Remediation")

idle_df = get_idle_ec2_instances()

if not idle_df.empty:

    st.warning(f"{len(idle_df)} idle EC2 instances found")

    st.dataframe(idle_df)

    selected = st.multiselect(
        "Select instances to stop",
        idle_df["InstanceId"].tolist()
    )

    if st.button("🚀 Stop Selected Instances"):

        if selected:
            msg = stop_instances(selected)
            st.success(msg)
        else:
            st.warning("Please select instances")

else:
    st.success("No idle EC2 instances found")

# ---------- ANOMALY ----------
st.subheader("⚠️ AI Cost Anomaly Detection")

anomalies = detect_anomaly(trend_df)

if not anomalies.empty:
    st.error("Anomaly Detected!")
    st.dataframe(anomalies)
else:
    st.success("No anomalies detected")

# ---------- FORECAST ----------
st.subheader("🔮 Forecast")

forecast = forecast_cost(trend_df)
st.metric("Next Month Prediction", f"${forecast}")

# ---------- RECOMMENDATIONS ----------
st.subheader("💡 AI Recommendations")

recs = generate_recommendations(service_df)

for r in recs:
    st.write("•", r)

# ---------- CHAT ----------
st.subheader("💬 FinOps Copilot")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask anything about AWS cost...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        resp, tools = invoke_agent(prompt)

    st.session_state.messages.append({"role": "assistant", "content": resp})

    if tools:
        with st.expander("🔧 Tools Used"):
            for t in tools:
                st.json(t)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Copilot Info")

    st.code(st.session_state.session_id)

    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()