import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import sys
import os
from datetime import datetime
sys.path.append('.')

def kpi_anomalies(
    kpi_name: str,
    site_id: Optional[str] = None,
    sector_id: Optional[str] = None,
    start_date: Optional[str] = '2024-01-01',
    end_date: Optional[str] = '2024-02-29', ) -> str:
    """
    Summarizes anomalies for a KPI: total count, average values.

    Inputs:
    - kpi_name: Name of KPI (e.g., 'DL_Throughput')
    - site_id: Optional Site ID (e.g., 'SITE_001')
    - sector_id: Optional Sector ID (e.g., 'SITE_001_SECTOR_E')
    - start_date, end_date: Optional range ('YYYY-MM-DD' or 'DD.MM.YY')
    
    site id's are always formated as "SITE_001" only.
    sector id's are always formated as "SITE_001_SECTOR_A" only.
    Start Date is 2024-01-01
    Last Date is 2024-02-29

    
    Valid KPI columns include:
    RSRP, SINR, DL_Throughput, RTT, UL_Throughput, CPU_Utilization,
    Call_Drop_Rate, Active_Users, Handover_Success_Rate, Packet_Loss.

    Output:
    - Number of anomalies
    - Average value of KPI overall
    - Average of anomaly points above and below KPI average
    - Group classification (e.g., signal, throughput)
    """
    try:
        df = pd.read_csv("Data/df_ensemble.csv")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Filter
        df = df[df["KPI"] == kpi_name]
        if site_id:
            df = df[df["Site_ID"] == site_id]
        if sector_id:
            df = df[df["Sector_ID"] == sector_id]
        if start_date:
            df = df[df["Date"] >= pd.to_datetime(start_date, errors="coerce")]
        if end_date:
            df = df[df["Date"] <= pd.to_datetime(end_date, errors="coerce")]
            
            
        peak_anomaly_note = ""
        df_peak_check = df[df["KPI"] == kpi_name]

        # Group by Date
        daily_counts = df_peak_check.groupby("Date").size().reset_index(name="Anomaly_Count")
        if not daily_counts.empty:
            peak_day = daily_counts.loc[daily_counts["Anomaly_Count"].idxmax()]
            peak_anomaly_note = (
                f"Most anomalies for `{kpi_name}` occurred on **{peak_day['Date'].date()}** "
                f"with **{int(peak_day['Anomaly_Count'])} anomalies**.\n\n"
            )

        if df.empty:
            return f"No anomaly data found for `{kpi_name}` with given filters."

        # Load base KPI data to get overall KPI values
        df_base = pd.read_csv("Data/KPI_data_cleaned.csv")
        df_base["Date"] = pd.to_datetime(df_base["Date"], errors="coerce")

        # Filter to match anomaly filters
        df_base = df_base.dropna(subset=[kpi_name])
        if site_id:
            df_base = df_base[df_base["Site_ID"] == site_id]
        if sector_id:
            df_base = df_base[df_base["Sector_ID"] == sector_id]
        if start_date:
            df_base = df_base[df_base["Date"] >= pd.to_datetime(start_date, errors="coerce")]
        if end_date:
            df_base = df_base[df_base["Date"] <= pd.to_datetime(end_date, errors="coerce")]

        if df_base.empty:
            return "No base KPI data found to compare anomalies."

        kpi_avg = df_base[kpi_name].mean()
        anomalies = df.copy()
        anomaly_values = anomalies[kpi_name].dropna()

        above_avg = anomaly_values[anomaly_values > kpi_avg]
        below_avg = anomaly_values[anomaly_values <= kpi_avg]

        anomaly_count = len(anomaly_values)
        avg_above = above_avg.mean() if not above_avg.empty else "N/A"
        avg_below = below_avg.mean() if not below_avg.empty else "N/A"

        # KPI Grouping
        kpi_groups = {
            "signal": ["RSRP", "SINR"],
            "throughput": ["DL_Throughput", "UL_Throughput"],
            "latency": ["RTT"],
            "access": ["Active_Users", "CPU_Utilization", "Handover_Success_Rate"],
            "stability": ["Packet_Loss", "Call_Drop_Rate"]
        }

        group = "unknown"
        for g, kpis in kpi_groups.items():
            if kpi_name in kpis:
                group = g
                Kpis = kpi_groups[g]
                break
        
        rel_kpi = ["RSRP", "SINR", "DL_Throughput", "RTT", "UL_Throughput", "CPU_Utilization","Call_Drop_Rate", "Active_Users", "Handover_Success_Rate", "Packet_Loss"]
        
        related_kpi_summary = ""
        related_kpis = [k for k in rel_kpi if k != kpi_name]
        related_counts = {}

        # Get all dates where the main KPI had anomalies
        main_dates = set(df["Date"])

        # Reload full anomaly dataset (not just filtered one)
        df_full = pd.read_csv("Data/df_ensemble.csv")
        df_full["Date"] = pd.to_datetime(df_full["Date"], errors="coerce")

        # Apply same site/sector filtering if given
        if site_id:
            df_full = df_full[df_full["Site_ID"] == site_id]
        if sector_id:
            df_full = df_full[df_full["Sector_ID"] == sector_id]

        # Filter to only dates matching main KPI anomalies
        df_same_days = df_full[df_full["Date"].isin(main_dates)]

        for rkpi in related_kpis:
            count = df_same_days[df_same_days["KPI"] == rkpi].shape[0]
            if count > 0:
                related_counts[rkpi] = count

        if related_counts:
            related_kpi_summary = "On the same days, anomalies also occurred in:\n"
            for k, v in related_counts.items():
                related_kpi_summary += f"- `{k}`: {v} times\n"
            

        return (
            f"**Anomaly Summary for KPI: `{kpi_name}`**\n\n"
            f"**Date Range Analyzed**: Based on filtered inputs or inferred from peak anomaly activity.\n\n"
            f"**Key Statistics**:\n"
            f"- Total anomalies detected: **{anomaly_count}**\n"
            f"- Baseline average of `{kpi_name}`: **{kpi_avg:.2f}**\n"
            f"- Avg anomaly value **above** baseline: **{avg_above if avg_above != 'N/A' else 'None'}**\n"
            f"- Avg anomaly value **below** baseline: **{avg_below if avg_below != 'N/A' else 'None'}**\n\n"
            
            f"**KPI Group Classification**:\n"
            f"- Group: **{group.capitalize()}**\n"
            f"- Related KPIs: {', '.join(Kpis)}\n\n"
            
            f"**Peak Anomaly Info**:\n"
            f"{peak_anomaly_note}"
            
            f"**Co-occurring Anomalies**:\n"
            f"{related_kpi_summary}\n"

            f"**LLM Usage Instructions**:\n"
            f"- Use this summary to respond to questions about `{kpi_name}` trends, anomaly patterns, and relationships.\n"
            f"- If a user asks about **why** anomalies occurred, do **not** generate speculative causes.\n"
            f"- Instead, guide the user toward requesting a root cause analysis.\n"
        )
    except Exception as e:
        return f"Error analyzing KPI anomalies: {str(e)}"



def test_basic_dl_throughput():
    print("Test 1: DL_Throughput anomaly summary, full dataset")
    result = kpi_anomalies(
        kpi_name="DL_Throughput"
    )
    print(result)
    print("-" * 80)

def test_with_site():
    print("Test 2: DL_Throughput, with site filter")
    result = kpi_anomalies(
        kpi_name="DL_Throughput",
        site_id="SITE_010"
    )
    print(result)
    print("-" * 80)

def test_with_sector_and_dates():
    print("Test 3: DL_Throughput, sector and date filter")
    result = kpi_anomalies(
        kpi_name="DL_Throughput",
        site_id="SITE_024",
        sector_id="SITE_024_SECTOR_E",
        start_date="2024-01-15",
        end_date="2024-02-10"
    )
    print(result)
    print("-" * 80)

def test_invalid_kpi():
    print("Test 4: Invalid KPI")
    result = kpi_anomalies(kpi_name="INVALID_KPI")
    print(result)
    print("-" * 80)

if __name__ == "__main__":
    test_basic_dl_throughput()
    test_with_site()
    test_with_sector_and_dates()
    test_invalid_kpi()

# import requests
# from langchain_core.messages import HumanMessage, AIMessage, messages_to_dict

# # Simulate ongoing conversation
# chat_history = [
#     HumanMessage(content="What is RSRP?"),
#     AIMessage(content="RSRP is a signal strength metric in LTE networks.")
# ]

# serialized_history = messages_to_dict(chat_history)

# payload = {
#     "input": "And what is SINR?",
#     "chat_history": serialized_history
# }

# res = requests.post("http://localhost:8000/invoke", json=payload)
# print(res.json())




# import requests

# res = requests.post("http://localhost:8000/invoke", json={
#     "input": "What is RSRP and why is it important in cellular networks?",
#     "chat_history": []
# })

# print(res.status_code)
# print(res.json())



# from Agent import agent_executor

# result = agent_executor.invoke({
#     "input": "What is RSRP?",
#     "chat_history": []
# })

# print(result)
