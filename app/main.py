"""
Streamlit dashboard for interactive exploration of the TCN Autoencoder's
anomaly detection results.

Panels:
  1. Signal Explorer       — multivariate channel selector with raw + reconstruction overlay
  2. Reconstruction Overlay — original vs reconstructed signal for a selected channel
  3. Anomaly Score Timeline — smoothed error with adjustable threshold slider
  4. Channel Contribution  — per-channel reconstruction error breakdown at anomaly events

Report:
  Generate Full Report — serializes current view data to results/streamlit_report.json
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
PROCESSED_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TCN Anomaly Detector",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading with Streamlit cache
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data artifacts...")
def load_artifacts():
    required = [
        RESULTS_DIR / "anomaly_scores.csv",
        RESULTS_DIR / "anomalies_percentile.csv",
        RESULTS_DIR / "anomalies_pot.csv",
        RESULTS_DIR / "reconstructions.npy",
        PROCESSED_DIR / "test.npy",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return None, "missing"

    scores_df = pd.read_csv(RESULTS_DIR / "anomaly_scores.csv")
    percentile_df = pd.read_csv(RESULTS_DIR / "anomalies_percentile.csv")
    pot_df = pd.read_csv(RESULTS_DIR / "anomalies_pot.csv")
    reconstructions = np.load(RESULTS_DIR / "reconstructions.npy")
    test_windows = np.load(PROCESSED_DIR / "test.npy")

    thresholds = {}
    thresholds_path = RESULTS_DIR / "thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            thresholds = json.load(f)

    return {
        "scores_df": scores_df,
        "percentile_df": percentile_df,
        "pot_df": pot_df,
        "reconstructions": reconstructions,
        "test_windows": test_windows,
        "thresholds": thresholds,
    }, None


def get_per_timestep_data(windows: np.ndarray) -> np.ndarray:
    """
    Convert windowed data back to a per-timestep representation by
    taking the last timestep from each window. This produces a sequence
    of length (num_windows,) for each channel.
    """
    return windows[:, -1, :]


def compute_channel_contributions(
    originals: np.ndarray, reconstructions: np.ndarray, timestep: int
) -> np.ndarray:
    """
    Per-channel MSE at a specific window index (anomaly event).
    Returns an array of shape (num_channels,).
    """
    orig = originals[timestep]
    recon = reconstructions[timestep]
    return np.mean((orig - recon) ** 2, axis=0)


def generate_report(data: dict, selected_channels: list, anomaly_idx: int) -> dict:
    test_windows = data["test_windows"]
    reconstructions = data["reconstructions"]
    scores_df = data["scores_df"]

    original_ts = get_per_timestep_data(test_windows)
    recon_ts = get_per_timestep_data(reconstructions)

    channel_contributions = compute_channel_contributions(
        test_windows, reconstructions, anomaly_idx
    )

    num_channels = test_windows.shape[2]
    channel_names = [f"channel_{i}" for i in range(num_channels)]

    report = {
        "signalData": {
            ch_name: original_ts[:, i].tolist()
            for i, ch_name in enumerate(channel_names)
            if i in selected_channels
        },
        "reconstructionData": {
            ch_name: recon_ts[:, i].tolist()
            for i, ch_name in enumerate(channel_names)
            if i in selected_channels
        },
        "anomalyScores": [
            {"timestamp": int(row["timestamp"]), "score": float(row["smoothed_error"])}
            for _, row in scores_df.iterrows()
        ],
        "channelContributions": {
            channel_names[i]: float(channel_contributions[i])
            for i in range(num_channels)
        },
    }

    report_path = RESULTS_DIR / "streamlit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def run_pipeline():
    import subprocess
    import sys
    try:
        with st.spinner("🚀 First run detected! Running preprocessing, training, and evaluation pipeline (this will take 10-15 minutes). Please don't close this page..."):
            subprocess.run([sys.executable, "scripts/preprocess_data.py"], check=True)
            subprocess.run([sys.executable, "scripts/train.py"], check=True)
            subprocess.run([sys.executable, "scripts/evaluate.py"], check=True)
        st.success("Pipeline completed successfully! Refreshing...")
        st.rerun()
    except subprocess.CalledProcessError as e:
        st.error(f"Pipeline failed: {e}")

# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main():
    st.title("Time-Series Anomaly Detector")
    st.markdown(
        "An interactive dashboard for exploring TCN Autoencoder reconstruction "
        "errors and anomaly detection results on NASA SMAP telemetry data."
    )
    st.markdown("---")

    data, error = load_artifacts()

    if error:
        st.warning("No trained model or processed data found. Running data pipeline automatically...")
        run_pipeline()
        return

    scores_df = data["scores_df"]
    percentile_df = data["percentile_df"]
    pot_df = data["pot_df"]
    test_windows = data["test_windows"]
    reconstructions = data["reconstructions"]
    thresholds = data["thresholds"]

    num_channels = test_windows.shape[2]
    channel_names = [f"Channel {i}" for i in range(num_channels)]

    # -----------------------------------------------------------------------
    # Sidebar controls
    # -----------------------------------------------------------------------
    st.sidebar.header("Controls")

    selected_channels = st.sidebar.multiselect(
        "Select channels to display",
        options=list(range(num_channels)),
        default=list(range(min(num_channels, 3))),
        format_func=lambda x: channel_names[x],
    )

    threshold_method = st.sidebar.radio(
        "Thresholding method",
        options=["Percentile", "POT (Peak-Over-Threshold)"],
        index=0,
    )

    score_min = float(scores_df["smoothed_error"].min())
    score_max = float(scores_df["smoothed_error"].max())
    default_threshold = float(
        thresholds.get("percentile_threshold", score_min + (score_max - score_min) * 0.9)
        if threshold_method == "Percentile"
        else thresholds.get("pot_threshold", score_min + (score_max - score_min) * 0.9)
    )
    default_threshold = max(score_min, min(score_max, default_threshold))

    manual_threshold = st.sidebar.slider(
        "Anomaly threshold",
        min_value=score_min,
        max_value=score_max,
        value=default_threshold,
        step=(score_max - score_min) / 500,
        format="%.6f",
    )

    # -----------------------------------------------------------------------
    # Panel 1: Signal Explorer
    # -----------------------------------------------------------------------
    st.subheader("Signal Explorer")

    if not selected_channels:
        st.warning("Select at least one channel from the sidebar.")
    else:
        original_ts = get_per_timestep_data(test_windows)

        fig_signal = go.Figure()
        for ch_idx in selected_channels:
            fig_signal.add_trace(
                go.Scatter(
                    y=original_ts[:, ch_idx],
                    mode="lines",
                    name=channel_names[ch_idx],
                    line=dict(width=1),
                )
            )

        fig_signal.update_layout(
            xaxis_title="Timestep",
            yaxis_title="Normalized Value",
            legend=dict(orientation="h"),
            height=350,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_signal, use_container_width=True)

    # -----------------------------------------------------------------------
    # Panel 2: Reconstruction Overlay
    # -----------------------------------------------------------------------
    st.subheader("Reconstruction Overlay")

    recon_channel = st.selectbox(
        "Select channel for reconstruction overlay",
        options=list(range(num_channels)),
        format_func=lambda x: channel_names[x],
    )

    original_ts = get_per_timestep_data(test_windows)
    recon_ts = get_per_timestep_data(reconstructions)

    fig_recon = go.Figure()
    fig_recon.add_trace(
        go.Scatter(
            y=original_ts[:, recon_channel],
            mode="lines",
            name="Original",
            line=dict(color="#1f77b4", width=1.5),
        )
    )
    fig_recon.add_trace(
        go.Scatter(
            y=recon_ts[:, recon_channel],
            mode="lines",
            name="Reconstruction",
            line=dict(color="#ff7f0e", width=1.5, dash="dash"),
        )
    )
    fig_recon.update_layout(
        xaxis_title="Timestep",
        yaxis_title="Normalized Value",
        legend=dict(orientation="h"),
        height=350,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_recon, use_container_width=True)

    # -----------------------------------------------------------------------
    # Panel 3: Anomaly Score Timeline
    # -----------------------------------------------------------------------
    st.subheader("Anomaly Score Timeline")

    dynamic_anomaly_mask = scores_df["smoothed_error"] > manual_threshold
    dynamic_anomaly_df = scores_df[dynamic_anomaly_mask]

    fig_score = go.Figure()
    fig_score.add_trace(
        go.Scatter(
            x=scores_df["timestamp"],
            y=scores_df["smoothed_error"],
            mode="lines",
            name="Smoothed Error",
            line=dict(color="#2ca02c", width=1.5),
        )
    )
    fig_score.add_hline(
        y=manual_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {manual_threshold:.4f}",
        annotation_position="top right",
    )
    if len(dynamic_anomaly_df) > 0:
        fig_score.add_trace(
            go.Scatter(
                x=dynamic_anomaly_df["timestamp"],
                y=dynamic_anomaly_df["smoothed_error"],
                mode="markers",
                name=f"Anomalies ({len(dynamic_anomaly_df)})",
                marker=dict(color="red", size=4, symbol="x"),
            )
        )
    fig_score.update_layout(
        xaxis_title="Timestep",
        yaxis_title="Reconstruction Error (smoothed)",
        legend=dict(orientation="h"),
        height=350,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_score, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total anomalies detected", len(dynamic_anomaly_df))
    col2.metric(
        "Percentile anomalies",
        len(percentile_df),
        help="Anomalies from the 99th percentile fixed threshold method",
    )
    col3.metric(
        "POT anomalies",
        len(pot_df),
        help="Anomalies from the Peak-Over-Threshold statistical method",
    )

    # -----------------------------------------------------------------------
    # Panel 4: Channel Contribution
    # -----------------------------------------------------------------------
    st.subheader("Channel Contribution at Anomaly Event")

    if len(dynamic_anomaly_df) > 0:
        anomaly_timestamps = sorted(dynamic_anomaly_df["timestamp"].tolist())
        selected_event = st.selectbox(
            "Select an anomaly event to inspect",
            options=anomaly_timestamps,
            format_func=lambda x: f"Timestep {x}",
        )

        window_idx = int(selected_event)
        if window_idx < len(test_windows):
            contributions = compute_channel_contributions(
                test_windows, reconstructions, window_idx
            )
            contrib_df = pd.DataFrame(
                {
                    "channel": channel_names,
                    "reconstruction_error": contributions,
                }
            ).sort_values("reconstruction_error", ascending=False)

            fig_contrib = go.Figure(
                go.Bar(
                    x=contrib_df["channel"],
                    y=contrib_df["reconstruction_error"],
                    marker_color="#9467bd",
                )
            )
            fig_contrib.update_layout(
                xaxis_title="Channel",
                yaxis_title="Per-Channel MSE",
                height=300,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_contrib, use_container_width=True)
            st.caption(
                f"Root cause analysis at timestep {selected_event}. "
                "Higher bar indicates the channel contributed more to the anomaly detection."
            )
    else:
        st.info(
            "No anomalies detected at the current threshold. "
            "Reduce the threshold slider to inspect channel contributions."
        )

    # -----------------------------------------------------------------------
    # Generate Full Report
    # -----------------------------------------------------------------------
    st.markdown("---")

    if st.button("Generate Full Report"):
        with st.spinner("Generating report..."):
            try:
                anomaly_event_idx = (
                    int(dynamic_anomaly_df["timestamp"].iloc[0])
                    if len(dynamic_anomaly_df) > 0
                    else 0
                )
                report = generate_report(
                    data,
                    selected_channels if selected_channels else list(range(num_channels)),
                    anomaly_event_idx,
                )
                st.success(
                    f"Report written to results/streamlit_report.json "
                    f"({len(report['anomalyScores'])} score entries, "
                    f"{len(report['channelContributions'])} channel contributions)"
                )
                with st.expander("Report preview (first 3 anomaly scores)"):
                    st.json(
                        {
                            "anomalyScores_sample": report["anomalyScores"][:3],
                            "channelContributions": report["channelContributions"],
                        }
                    )
            except Exception as exc:
                st.error(f"Report generation failed: {exc}")


if __name__ == "__main__":
    main()
