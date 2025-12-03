import pandas as pd
from datetime import datetime, timedelta
from langchain.tools import tool
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import os
import requests

KPI_CSV_PATH = "Data\KPI_data_cleaned.csv"

@tool(return_direct=True)
def get_site_kpi_extreme(
    kpi_name: str,
    extreme_type: str = "highest",
    start_date: str = None,
    end_date: str = None
) -> str:
    """
    Returns the site with the highest or lowest average value for the given KPI 
    (e.g., SINR, DL_Throughput, etc.) within a specified date range.

    Parameters:
    - kpi_name: Name of the KPI column (e.g., "SINR", "DL_Throughput").
    - extreme_type: "highest" or "lowest" (default: "highest").
    - start_date, end_date: Optional date range in "YYYY-MM-DD" or "DD.MM.YY" format.
    
    Start Date is 2024-01-01
    Last Date is 2024-02-29

    Valid KPI columns include:
    RSRP, SINR, DL_Throughput, RTT, UL_Throughput, CPU_Utilization,
    Call_Drop_Rate, Active_Users, Handover_Success_Rate, Packet_Loss.
    """

    try:
        df = pd.read_csv(KPI_CSV_PATH)

        # Parse and clean Date column
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Site_ID", kpi_name])

        # Parse date strings
        if end_date:
            end_date = pd.to_datetime(end_date, errors="coerce")
        else:
            end_date = df["Date"].max()

        if start_date:
            start_date = pd.to_datetime(start_date, errors="coerce")
        else:
            start_date = end_date - timedelta(days=7)

        # Filter by date range
        filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

        if filtered.empty:
            return f"No data available for `{kpi_name}` between {start_date.date()} and {end_date.date()}."

        # Calculate average KPI by site
        avg_kpi = filtered.groupby("Site_ID")[kpi_name].mean()

        if extreme_type.lower() == "lowest":
            site_id = avg_kpi.idxmin()
            value = avg_kpi.min()
            direction = "lowest"
        else:
            site_id = avg_kpi.idxmax()
            value = avg_kpi.max()
            direction = "highest"

        return (
            f"Between **{start_date.date()}** and **{end_date.date()}**, "
            f"site `{site_id}` had the {direction} average **{kpi_name}** of **{value:.2f}**."
        )

    except Exception as e:
        return f"Error processing KPI data: {str(e)}"

@tool(return_direct=True)
def get_peak_kpi_day_for_site(
    site_id: str,
    kpi_name: str = "DL_Throughput",
    extreme_type: str = "highest",
    start_date: str = None,
    end_date: str = None
) -> str:
    """
    Returns the day on which a given site experienced the highest or lowest value of a KPI.
    
    Parameters:
    - site_id: e.g., "SITE_001"
    - kpi_name: e.g., "DL_Throughput", "SINR", "RTT" (default is "DL_Throughput")
    - extreme_type: "highest" or "lowest" (default is "highest")
    - start_date, end_date: Optional; in "YYYY-MM-DD" or "DD.MM.YY" format
    
    
    site id's are always formated as "SITE_001" only.
    Start Date is 2024-01-01
    Last Date is 2024-02-29

    
    Valid KPI columns include:
    RSRP, SINR, DL_Throughput, RTT, UL_Throughput, CPU_Utilization,
    Call_Drop_Rate, Active_Users, Handover_Success_Rate, Packet_Loss.

    Example:
    "On which day did SITE_001 have the highest DL_Throughput?"
    """

    try:
        df = pd.read_csv(KPI_CSV_PATH)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Site_ID", kpi_name])

        # Filter for the specified site
        site_df = df[df["Site_ID"] == site_id]
        if site_df.empty:
            return f"No data found for site `{site_id}`."

        # Handle date filtering
        if end_date:
            end_date = pd.to_datetime(end_date, errors="coerce")
        else:
            end_date = site_df["Date"].max()

        if start_date:
            start_date = pd.to_datetime(start_date, errors="coerce")
        else:
            start_date = end_date - timedelta(days=30)

        site_df = site_df[(site_df["Date"] >= start_date) & (site_df["Date"] <= end_date)]

        if site_df.empty:
            return f"No data found for `{site_id}` between {start_date.date()} and {end_date.date()}."

        # Find the day with highest or lowest KPI
        if extreme_type.lower() == "lowest":
            peak_row = site_df.loc[site_df[kpi_name].idxmin()]
            label = "lowest"
        else:
            peak_row = site_df.loc[site_df[kpi_name].idxmax()]
            label = "highest"

        date = peak_row["Date"].date()
        value = peak_row[kpi_name]

        return (
            f" On **{date}**, site `{site_id}` had the {label} **{kpi_name}** of **{value:.2f}** "
            f"between {start_date.date()} and {end_date.date()}."
        )

    except Exception as e:
        return f"Error processing request: {str(e)}"
    
@tool
def compare_kpi_impact(
    kpi_x: str,
    kpi_y: str,
    site_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """
    Estimates how often an increase in KPI_X is followed by an increase in KPI_Y,
    and performs a Granger Causality test to assess if KPI_X helps predict KPI_Y.
    
    site id's are always formated as "SITE_001" only.
    Start Date is 2024-01-01
    Last Date is 2024-02-29

    
    Valid KPI columns include:
    RSRP, SINR, DL_Throughput, RTT, UL_Throughput, CPU_Utilization,
    Call_Drop_Rate, Active_Users, Handover_Success_Rate, Packet_Loss.
    
    Example: Does an increase in Active_Users lead to an increase in CPU_Utilization?
    """
    try:
        df = pd.read_csv(KPI_CSV_PATH)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Site_ID", kpi_x, kpi_y])

        if site_id:
            df = df[df["Site_ID"] == site_id]

        if start_date:
            df = df[df["Date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["Date"] <= pd.to_datetime(end_date)]

        if df.empty:
            return f"No data available for {kpi_x} and {kpi_y}."

        df = df.sort_values("Date")
        df = df[[kpi_x, kpi_y]].dropna()

        # Directional probability logic
        df["ΔX"] = df[kpi_x].diff()
        df["ΔY"] = df[kpi_y].diff()

        valid = df.dropna(subset=["ΔX", "ΔY"])
        rising_x = valid[valid["ΔX"] > 0]
        rising_both = rising_x[rising_x["ΔY"] > 0]

        if len(rising_x) == 0:
            directional_comment = (
                f"No positive changes in {kpi_x} to evaluate directional effect on {kpi_y}."
            )
            directional_ratio = 0
        else:
            conditional_prob = len(rising_both) / len(rising_x)
            directional_ratio = conditional_prob
            directional_comment = (
                f"When **{kpi_x} increases**, **{kpi_y} increases** "
                f"{conditional_prob * 100:.1f}% of the time over the selected period "
                f"{'for site `' + site_id + '`' if site_id else '(all sites)'}. "
                f"This suggests a {'likely' if conditional_prob > 0.45 else 'weak'} directional relationship."
            )

        # Granger causality test (maxlag=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_gc = df[[kpi_y, kpi_x]].dropna()  # Granger expects Y first
            result = grangercausalitytests(df_gc, maxlag=2, verbose=False)

        best_p_value = min([result[lag][0]["ssr_ftest"][1] for lag in result])
        granger_comment = (
            f"**Granger Causality Test**: Examining whether changes in `{kpi_x}` help predict future changes in `{kpi_y}`.\n"
            f"→ The p-value is **{best_p_value:.4f}**.\n"
            + (
                f"Since the p-value is less than 0.05, this suggests that changes in **`{kpi_x}` likely help predict future values of `{kpi_y}`** (i.e., `{kpi_x}` Granger-causes `{kpi_y}`).\n"
                if best_p_value < 0.05 else
                f"Since the p-value is greater than 0.05, there is **no statistical evidence** that `{kpi_x}` helps predict `{kpi_y}`.\n"
                
            )

        )

        return directional_comment + "\n\n" + granger_comment + f"Answer the quetion with general knowedge with the above results as proof with numbers."

    except Exception as e:
        return f"Error evaluating directional KPI impact: {str(e)}"

@tool
def describe_kpi_dataset(dummy_input: Optional[str] = None) -> str:
    """
    Provides a summary of the available KPI dataset, including:
    - Date range
    - Number of sites and sectors
    - List of KPIs
    - Missing values, mean, min, and max per KPI
    """
    try:
        df = pd.read_csv(KPI_CSV_PATH)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        if df.empty:
            return "Dataset appears empty or could not be parsed."

        date_range = f"{df['Date'].min().date()} to {df['Date'].max().date()}"
        num_sites = df["Site_ID"].nunique()
        num_sectors = df["Sector_ID"].nunique() if "Sector_ID" in df.columns else "N/A"

        kpi_columns = [col for col in df.columns if col not in ["Date", "Site_ID", "Sector_ID"]]
        kpi_stats = df[kpi_columns].agg(['mean', 'min', 'max']).transpose()
        missing_counts = df[kpi_columns].isnull().sum().to_dict()

        summary = (
            f"**KPI Dataset Overview**\n"
            f"- Date Range: **{date_range}**\n"
            f"- Sites: **{num_sites}**\n"
            f"- Sectors: **{num_sectors}**\n"
            f"- Available KPIs ({len(kpi_columns)}): {', '.join(kpi_columns)}\n\n"
            f"**Per-KPI Statistics**\n"
        )

        for kpi in kpi_columns:
            summary += (
                f"• **{kpi}**\n"
                f"  - Missing values: {missing_counts.get(kpi, 0)}\n"
                f"  - Mean: {kpi_stats.loc[kpi, 'mean']:.2f}\n"
                f"  - Min: {kpi_stats.loc[kpi, 'min']:.2f}\n"
                f"  - Max: {kpi_stats.loc[kpi, 'max']:.2f}\n"
            )

        return summary.strip()

    except Exception as e:
        return f"Failed to describe KPI dataset: {str(e)}"

    
    
@tool
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
            "latency": ["DL_Throughput", "UL_Throughput", "RTT"],
            "access": ["Active_Users", "CPU_Utilization", "Handover_Success_Rate"],
            "stability": ["Packet_Loss", "Call_Drop_Rate"]
        }

        group = "unknown"
        for g, kpis in kpi_groups.items():
            if kpi_name in kpis:
                group = g
                Kpis = kpi_groups[g]
                break
        
        
        related_kpi_summary = ""
        related_kpis = [k for k in Kpis if k != kpi_name]
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
            related_kpi_summary = "On the same days, anomalies also occurred in other KPI's which may be cause for anomalies in {kpi_name}:\n"
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
            f"- Use this above summary about `{kpi_name}` trends, anomaly patterns, and relationships to answer with details.\n"
            f"- If a user asks about **why** anomalies occurred, do **not** generate speculative causes. tell the answer based on the related  kpi summary.\n"
        )
    except Exception as e:
        return f"Error analyzing KPI anomalies: {str(e)}"
