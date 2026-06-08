"""
BDD100K ADAS Analytics Dashboard.

This module provides a Streamlit-based interactive dashboard for analyzing
ADAS dataset statistics, distributions, spatial priors, and edge cases.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import entropy

# =============================================================================
# Path Configuration (Docker & GitHub Friendly)
# =============================================================================
# Defaults to current working directory, can be overridden by environment variables
BASE_DIR = Path(os.getenv("APP_BASE_DIR", "."))

TRAIN_IMG_DIR = BASE_DIR / "data" / "bdd100k_images_100k" / "bdd100k" / "images" / "100k" / "train"
VAL_IMG_DIR = BASE_DIR / "data" / "bdd100k_images_100k" / "bdd100k" / "images" / "100k" / "val"

TRAIN_REPORT_PATH = "./analysis_reports/train_report.json"
VAL_REPORT_PATH = "./analysis_reports/val_report.json"

# =============================================================================
# Page Configuration & Styling
# =============================================================================
st.set_page_config(
    page_title="BDD100K ADAS Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a polished, dark-themed UI
st.markdown(
    """
    <style>
    .main { 
        background-color: #0e1117; 
    }
    .stMetric {
        border-radius: 12px;
        padding: 15px;
        background-color: #1f2937;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border: 1px solid #374151;
    }
    h1, h2, h3 { 
        color: #f8f9fa; 
        font-weight: 600;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Data Loading & Image Rendering
# =============================================================================
@st.cache_data
def load_report(filepath: str) -> Optional[Dict[str, Any]]:
    """Load JSON report data from the given filepath."""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return None


def render_image_with_boxes(
    image_name: str,
    boxes: List[List[float]],
    categories: List[str],
    overlay_text: str = ""
) -> np.ndarray:
    """Load an image and draw bounding boxes using OpenCV."""
    img_path = os.path.join(TRAIN_IMG_DIR, image_name)
    if not os.path.exists(img_path):
        img_path = os.path.join(VAL_IMG_DIR, image_name)

    if not os.path.exists(img_path):
        placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(placeholder, f"Image not found locally: {image_name}", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return placeholder

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for box, category in zip(boxes, categories):
        if not box or len(box) != 4: continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (59, 130, 246), 2)
        (text_w, text_h), _ = cv2.getTextSize(category, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + text_w, y1), (59, 130, 246), -1)
        cv2.putText(img, category, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if overlay_text:
        (tw, th), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(img, (20, 20), (40 + tw, 40 + th), (0, 0, 0), -1)
        cv2.putText(img, overlay_text, (30, 30 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    return img


train_data = load_report(TRAIN_REPORT_PATH)
val_data = load_report(VAL_REPORT_PATH)

if not train_data:
    st.error(f"⚠️ Report not found at '{TRAIN_REPORT_PATH}'. Please run analyzer first.")
    st.stop()


# =============================================================================
# Sidebar Navigation & Global Filters
# =============================================================================
st.sidebar.title("🚘 ADAS Analytics")
st.sidebar.markdown("---")

PAGES = [
    "Overview & Metadata",
    "Class & Geometry Analysis",
    "Train vs Validation",
    "Co-occurrence Analysis",
    "Anomalies & Edge Cases",
    "Interesting & Hard Samples",
]
selection = st.sidebar.radio("Navigation", PAGES)

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Global Filters")
st.sidebar.caption("Applies to Hard Samples Explorer.")

meta_dist = train_data.get("metadata_distribution", {})
available_weather = list(meta_dist.get("weather", {}).keys())
available_time = list(meta_dist.get("timeofday", {}).keys())

selected_weather = st.sidebar.multiselect("Weather", available_weather, default=available_weather)
selected_time = st.sidebar.multiselect("Time of Day", available_time, default=available_time)


def get_class_df(data: Dict[str, Any]) -> pd.DataFrame:
    dist = data.get("class_distribution", {})
    df = pd.DataFrame.from_dict(dist, orient="index").reset_index()
    df.rename(columns={"index": "Class", "count": "Instances", "images": "Images"}, inplace=True)
    df["Instances per Image"] = (df["Instances"] / df["Images"].replace(0, 1)).round(2)
    return df.sort_values(by="Instances", ascending=False)


# =============================================================================
# Main Pages
# =============================================================================

if selection == "Overview & Metadata":
    st.title("📊 Dataset Overview & Metadata")
    summary = train_data.get("summary", {})
    df_classes = get_class_df(train_data)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Frames", f"{summary.get('total_frames', 0):,}")
    col2.metric("Total Objects", f"{summary.get('total_objects', 0):,}")
    col3.metric("Total Classes", len(df_classes))
    col4.metric("Avg Objects / Frame", f"{summary.get('avg_objects_per_frame', 0):.1f}")
    st.markdown("---")
    
    col_tree, col_pareto = st.columns(2)
    with col_tree:
        st.subheader("Dataset Composition")
        fig_tree = px.treemap(df_classes, path=["Class"], values="Instances", color="Instances", color_continuous_scale="Blues", template="plotly_dark")
        st.plotly_chart(fig_tree, use_container_width=True)

    with col_pareto:
        st.subheader("Class Imbalance (Pareto)")
        df_classes["cumulative_pct"] = (df_classes["Instances"].cumsum() / df_classes["Instances"].sum() * 100)
        fig_pareto = go.Figure()
        # ADDED TEXT PROPERTY HERE
        fig_pareto.add_trace(go.Bar(x=df_classes["Class"], y=df_classes["Instances"], name="Annotations", marker_color="#3b82f6", text=df_classes["Instances"], textposition="outside"))
        fig_pareto.add_trace(go.Scatter(x=df_classes["Class"], y=df_classes["cumulative_pct"], name="Cumulative %", yaxis="y2", mode="lines+markers", line=dict(color="#f59e0b", width=3)))
        fig_pareto.update_layout(template="plotly_dark", yaxis=dict(title="Annotations"), yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]), showlegend=False)
        st.plotly_chart(fig_pareto, use_container_width=True)

    st.markdown("---")
    st.subheader("🌦️ Environmental Metadata")
    meta = train_data.get("metadata_distribution", {})
    c1, c2, c3 = st.columns(3)

    with c1:
        weather = pd.Series(meta.get("weather", {})).reset_index()
        weather.columns = ["Weather", "Count"]
        st.plotly_chart(px.pie(weather, values="Count", names="Weather", hole=0.4, template="plotly_dark", title="Weather"), use_container_width=True)

    with c2:
        tod = pd.Series(meta.get("timeofday", {})).reset_index()
        tod.columns = ["Time", "Count"]
        st.plotly_chart(px.pie(tod, values="Count", names="Time", hole=0.4, template="plotly_dark", title="Time of Day"), use_container_width=True)

    with c3:
        scene = pd.Series(meta.get("scene", {})).reset_index()
        scene.columns = ["Scene", "Count"]
        # ADDED TEXT PROPERTY HERE
        fig_scene = px.bar(scene, x="Scene", y="Count", template="plotly_dark", title="Scene Type", text="Count")
        fig_scene.update_traces(textposition="outside")
        st.plotly_chart(fig_scene, use_container_width=True)

elif selection == "Class & Geometry Analysis":
    st.title("🏷️ Class & Geometry Analysis")
    df_classes = get_class_df(train_data)

    st.subheader("Per-Class Statistics")
    st.dataframe(df_classes.style.background_gradient(cmap="Blues", subset=["Instances", "Images", "Instances per Image"]), use_container_width=True)
    st.markdown("---")
    
    st.subheader("Object Size Distribution by Class")
    size_dist = train_data.get("size_distribution", {})
    if size_dist:
        df_size = pd.DataFrame.from_dict(size_dist, orient="index").fillna(0)
        # ADDED TEXT_AUTO HERE for stacked bars
        fig_size = px.bar(df_size, barmode="stack", template="plotly_dark", color_discrete_sequence=["#60a5fa", "#34d399", "#f87171"], text_auto=True)
        st.plotly_chart(fig_size, use_container_width=True)

    st.markdown("---")
    st.subheader("📐 Spatial & Geometric Priors")
    geo_samples = train_data.get("geometry_samples", [])

    if geo_samples:
        df_geo = pd.DataFrame(geo_samples)
        st.caption("Displays the 2D histogram of object centers. Useful for ADAS ground-plane calibration checks.")
        if "center_x" in df_geo.columns and "center_y" in df_geo.columns:
            fig_spatial = px.density_heatmap(df_geo, x="center_x", y="center_y", facet_col="category", facet_col_wrap=3, template="plotly_dark", color_continuous_scale="Inferno", range_x=[0, 1280], range_y=[720, 0])
            st.plotly_chart(fig_spatial, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.box(df_geo, x="category", y="area_ratio", template="plotly_dark", color="category", title="Area Distribution"), use_container_width=True)
        with col2:
            st.plotly_chart(px.violin(df_geo, x="category", y="aspect_ratio", box=True, template="plotly_dark", color="category", title="Aspect Ratio Distribution"), use_container_width=True)

elif selection == "Train vs Validation":
    st.title("⚖️ Train vs Validation Drift")

    if not val_data:
        st.warning("Validation report not found. Run analyzer on val split to view.")
    else:
        st.markdown("### Dataset Split Consistency")
        st.write(
            "Comparing percentage distributions between Train and Validation datasets."
        )

        # ============================================================
        # CLASS DISTRIBUTION (%)
        # ============================================================
        train_cls = get_class_df(train_data)[["Class", "Instances"]]
        val_cls = get_class_df(val_data)[["Class", "Instances"]]

        train_cls.rename(
            columns={"Class": "Category", "Instances": "Count"},
            inplace=True
        )

        val_cls.rename(
            columns={"Class": "Category", "Instances": "Count"},
            inplace=True
        )

        train_cls["Percentage"] = (
            train_cls["Count"] / train_cls["Count"].sum() * 100
        ).round(2)

        val_cls["Percentage"] = (
            val_cls["Count"] / val_cls["Count"].sum() * 100
        ).round(2)

        train_cls["Split"] = "Train"
        val_cls["Split"] = "Validation"

        df_cls = pd.concat([train_cls, val_cls])

        sort_order = (
            train_cls.sort_values("Percentage", ascending=False)
            ["Category"]
            .tolist()
        )

        df_cls["Category"] = pd.Categorical(
            df_cls["Category"],
            categories=sort_order,
            ordered=True,
        )

        st.subheader("Class Distribution (%)")

        fig_cls = px.bar(
            df_cls,
            x="Category",
            y="Percentage",
            color="Split",
            barmode="group",
            text="Percentage",
            color_discrete_sequence=["#4C78A8", "#E45756"],
        )

        fig_cls.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig_cls.update_layout(
            template="plotly_dark",
            yaxis_title="Percentage (%)"
        )

        st.plotly_chart(fig_cls, use_container_width=True)

        # ============================================================
        # WEATHER DISTRIBUTION (%)
        # ============================================================
        train_w = train_data.get(
            "metadata_distribution", {}
        ).get("weather", {})

        val_w = val_data.get(
            "metadata_distribution", {}
        ).get("weather", {})

        df_w_train = pd.DataFrame(
            list(train_w.items()),
            columns=["Weather", "Count"]
        )

        df_w_val = pd.DataFrame(
            list(val_w.items()),
            columns=["Weather", "Count"]
        )

        df_w_train["Percentage"] = (
            df_w_train["Count"] /
            df_w_train["Count"].sum() * 100
        ).round(2)

        df_w_val["Percentage"] = (
            df_w_val["Count"] /
            df_w_val["Count"].sum() * 100
        ).round(2)

        df_w_train["Split"] = "Train"
        df_w_val["Split"] = "Validation"

        df_weather = pd.concat([df_w_train, df_w_val])

        sort_order_weather = (
            df_w_train.sort_values(
                "Percentage",
                ascending=False
            )["Weather"].tolist()
        )

        df_weather["Weather"] = pd.Categorical(
            df_weather["Weather"],
            categories=sort_order_weather,
            ordered=True,
        )

        st.subheader("Weather Distribution (%)")

        fig_weather = px.bar(
            df_weather,
            x="Weather",
            y="Percentage",
            color="Split",
            barmode="group",
            text="Percentage",
            color_discrete_sequence=["#4C78A8", "#E45756"],
        )

        fig_weather.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig_weather.update_layout(
            template="plotly_dark",
            yaxis_title="Percentage (%)"
        )

        st.plotly_chart(fig_weather, use_container_width=True)

        # ============================================================
        # SCENE DISTRIBUTION (%)
        # ============================================================
        train_s = train_data.get(
            "metadata_distribution", {}
        ).get("scene", {})

        val_s = val_data.get(
            "metadata_distribution", {}
        ).get("scene", {})

        df_s_train = pd.DataFrame(
            list(train_s.items()),
            columns=["Scene", "Count"]
        )

        df_s_val = pd.DataFrame(
            list(val_s.items()),
            columns=["Scene", "Count"]
        )

        df_s_train["Percentage"] = (
            df_s_train["Count"] /
            df_s_train["Count"].sum() * 100
        ).round(2)

        df_s_val["Percentage"] = (
            df_s_val["Count"] /
            df_s_val["Count"].sum() * 100
        ).round(2)

        df_s_train["Split"] = "Train"
        df_s_val["Split"] = "Validation"

        df_scene = pd.concat([df_s_train, df_s_val])

        sort_order_scene = (
            df_s_train.sort_values(
                "Percentage",
                ascending=False
            )["Scene"].tolist()
        )

        df_scene["Scene"] = pd.Categorical(
            df_scene["Scene"],
            categories=sort_order_scene,
            ordered=True,
        )

        st.subheader("Scene Distribution (%)")

        fig_scene = px.bar(
            df_scene,
            x="Scene",
            y="Percentage",
            color="Split",
            barmode="group",
            text="Percentage",
            color_discrete_sequence=["#4C78A8", "#E45756"],
        )

        fig_scene.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig_scene.update_layout(
            template="plotly_dark",
            yaxis_title="Percentage (%)"
        )

        st.plotly_chart(fig_scene, use_container_width=True)

        # ============================================================
        # KL DIVERGENCE
        # ============================================================
        st.markdown("---")
        st.subheader("Distribution Drift Analytics")

        df_merged = pd.merge(
            train_cls[["Category", "Percentage"]],
            val_cls[["Category", "Percentage"]],
            on="Category",
            suffixes=("_train", "_val"),
        ).fillna(0)

        df_merged["pct_train"] = (
            df_merged["Percentage_train"] / 100
        )

        df_merged["pct_val"] = (
            df_merged["Percentage_val"] / 100
        )

        kl_div = entropy(
            df_merged["pct_train"],
            df_merged["pct_val"]
        )

        st.metric(
            "KL Divergence (Train || Val)",
            f"{kl_div:.8f}",
            help="Lower is better. Measures dataset drift."
        )

        # Optional Drift Table
        st.subheader("Per-Class Drift")

        df_merged["Drift (%)"] = (
            df_merged["Percentage_val"] -
            df_merged["Percentage_train"]
        ).round(2)

        st.dataframe(
            df_merged.sort_values(
                by="Drift (%)",
                key=abs,
                ascending=False
            ),
            use_container_width=True,
        )

elif selection == "Co-occurrence Analysis":
    st.title("🔗 Co-occurrence Analysis")
    co_oc = train_data.get("co_occurrence", {})
    if co_oc:
        st.plotly_chart(px.imshow(pd.DataFrame(co_oc).fillna(0), text_auto=True, aspect="auto", template="plotly_dark", color_continuous_scale="Viridis"), use_container_width=True)

elif selection == "Anomalies & Edge Cases":
    st.title("🚨 Anomalies & Edge Cases")
    st.markdown("### Bounding Box Geometry Anomalies")
    st.write("Analyzing object dimensions to detect potential labeling errors or physical limitations for camera sensors. Data represents a 5,000-point random subsample.")
    
    train_anom = train_data.get("anomalies", {})
    colA, colB = st.columns(2)
    with colA:
        st.info("🔬 **Micro-Boxes**\n\nObjects occupying less than 0.1% of the image area.")
        st.metric("Train Split Micro-Boxes", f"{train_anom.get('micro_boxes', 0):,}")
    with colB:
        st.warning("📐 **Extreme Aspect Ratios**\n\nWidth/Height ratio > 5.0 or < 0.2.")
        st.metric("Train Split Extreme Aspects", f"{train_anom.get('extreme_aspects', 0):,}")

    st.divider()
    geom_samples = train_data.get("geometry_samples", [])
    if geom_samples:
        df_geom = pd.DataFrame(geom_samples)
        fig_geom = px.scatter(df_geom, x="aspect_ratio", y="area_ratio", color="category", opacity=0.6, labels={"aspect_ratio": "Aspect Ratio (Width/Height)", "area_ratio": "Area Ratio (% of Image)"})
        fig_geom.add_vline(x=5.0, line_dash="dash", line_color="red", annotation_text="Extreme Width Anomaly (>5.0)", annotation_position="top right")
        fig_geom.add_vline(x=0.2, line_dash="dash", line_color="blue", annotation_text="Extreme Height Anomaly (<0.2)", annotation_position="top right")
        fig_geom.add_hline(y=0.001, line_dash="dash", line_color="orange", annotation_text="Micro-Box Anomaly (<0.1% Area)", annotation_position="bottom right")
        fig_geom.update_xaxes(range=[0, 8])
        st.plotly_chart(fig_geom, use_container_width=True)

elif selection == "Interesting & Hard Samples":
    st.title("🔍 Interesting & Hard Samples")
    tab1, tab2 = st.tabs(["📸 Edge Cases & Extremes", "🔥 Active Learning Explorer"])
    
    with tab1:
        samples = train_data.get("interesting_samples", {})
        sample_type = st.selectbox("Select Sample Type", ["largest_objects", "smallest_objects", "high_density_frames"])
        data_list = samples.get(sample_type, [])

        if data_list:
            df_samples = pd.DataFrame(data_list)
            st.dataframe(df_samples.head(10), use_container_width=True)
            st.subheader("Visualizer")
            '''
            frame_col = "name" if "name" in df_samples.columns else "frame"
            selected_row = st.selectbox("Select a frame to visualize:", df_samples[frame_col].unique())
            row_data = df_samples[df_samples[frame_col] == selected_row].iloc[0]
            '''
            selected_row = st.selectbox("Select a frame to visualize:",df_samples["name"].unique())

            row_data = df_samples[df_samples["name"] == selected_row].iloc[0]
            boxes = row_data.get("boxes", [])
            labels = row_data.get("labels", [])

            if sample_type == "largest_objects":
                cat = row_data.get('category', 'Unknown').upper()
                overlay = f"LARGEST {cat} | Area: {row_data.get('area_ratio', 0)*100:.1f}%"
            elif sample_type == "smallest_objects":
                cat = row_data.get('category', 'Unknown').upper()
                overlay = f"SMALLEST {cat} | Area: {row_data.get('area_ratio', 0)*100:.3f}%"
            else:
                objects_count = row_data.get('objects', 0)
                frame_labels = list(set(row_data.get('labels', [])))
                labels_str = ", ".join(frame_labels[:3])
                if len(frame_labels) > 3: labels_str += ", ..."
                overlay = f"DENSE FRAME: {objects_count} Objects ({labels_str})"

            if st.button("Load Edge Case Image"):
                img = render_image_with_boxes(selected_row, boxes, labels, overlay_text=overlay)
                st.image(img, caption=f"Frame: {selected_row}", use_container_width=True)

    with tab2:
        st.caption("Filter and identify frames where the environment is complex and model confidence might drop.")
        hard_samples = train_data.get("hard_samples", [])

        if hard_samples:
            df_hard = pd.DataFrame(hard_samples)
            df_filtered = df_hard[(df_hard["weather"].isin(selected_weather)) & (df_hard["timeofday"].isin(selected_time))]

            col1, col2 = st.columns(2)
            max_score = float(df_filtered["score"].max()) if not df_filtered.empty else 10.0
            min_hardness = col1.slider("Minimum Hardness Score", min_value=0.0, max_value=max_score, value=min(2.0, max_score))
            col2.slider("Max Model Confidence (Mocked feature for Active Learning)", min_value=0.0, max_value=1.0, value=0.4)

            df_filtered = df_filtered[df_filtered["score"] >= min_hardness]
            st.metric("Matching Hard Frames", len(df_filtered))
            st.dataframe(df_filtered.style.background_gradient(cmap="Reds", subset=["score"]), use_container_width=True, height=250)
            
            if not df_filtered.empty:
                st.subheader("Hard Sample Visualizer")
                hard_selected = st.selectbox("Select Hard Frame:", df_filtered["name"].unique())
                hard_row = df_filtered[df_filtered["name"] == hard_selected].iloc[0]
                
                if st.button("Load Hard Frame"):
                    overlay = f"Hardness: {hard_row['score']} | {str(hard_row['weather']).upper()} | {str(hard_row['timeofday']).upper()}"
                    img = render_image_with_boxes(hard_selected, [], [], overlay_text=overlay)
                    st.image(img, caption=f"Frame: {hard_selected}", use_container_width=True)