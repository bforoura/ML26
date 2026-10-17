import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import zoom
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="PCA Intuition: Camera Angle, Math & Applications",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Understanding PCA: From Intuition to Real-World Applications")
st.markdown(
    "Welcome to Principal Component Analysis! Use the navigation below to explore the physical camera analogy, the linear algebra mechanics, real-world high-dimensional dataset reduction, and image compression."
)

# -----------------------------------------------------------------------------
# Stateful Navigation
# -----------------------------------------------------------------------------
tabs_list = [
    "Tab 1: The Camera Angle Analogy",
    "Tab 2: Math & Variance",
    "Tab 3: Real Dataset Reduction",
    "Tab 4: Image Compression",
]

if "active_tab" not in st.session_state:
    st.session_state.active_tab = tabs_list[0]

selected_tab = st.radio(
    "Navigation Tabs",
    tabs_list,
    horizontal=True,
    key="active_tab",
)

st.markdown("---")

# -----------------------------------------------------------------------------
# High-Resolution 3D Point Cloud Generator for Detailed Cyclist Statue
# -----------------------------------------------------------------------------
@st.cache_data
def generate_highres_cyclist_statue(total_points=5000):
    np.random.seed(42)
    points = []

    def add_ring(center, radius, thickness, plane="xz", count=200):
        center = np.array(center)
        angles = np.random.uniform(0, 2 * np.pi, count)
        r = radius + np.random.normal(0, thickness, count)
        if plane == "xz":
            x = center[0] + r * np.cos(angles)
            y = center[1] + np.random.normal(0, thickness * 0.5, count)
            z = center[2] + r * np.sin(angles)
        elif plane == "xy":
            x = center[0] + r * np.cos(angles)
            y = center[1] + r * np.sin(angles)
            z = center[2] + np.random.normal(0, thickness * 0.5, count)
        return np.column_stack([x, y, z])

    def add_cylinder(p1, p2, radius, count=120):
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        t = np.random.uniform(0, 1, count)
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length == 0:
            return np.tile(p1, (count, 1))

        u_dir = direction / length
        ortho = np.array([1, 0, 0]) if abs(u_dir[0]) < 0.8 else np.array([0, 1, 0])
        v1 = np.cross(u_dir, ortho)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(u_dir, v1)

        angles = np.random.uniform(0, 2 * np.pi, count)
        r = np.random.uniform(0, radius, count)

        line_pts = p1 + np.outer(t, direction)
        radial_pts = (
            np.outer(r * np.cos(angles), v1) + np.outer(r * np.sin(angles), v2)
        )
        return line_pts + radial_pts

    def add_ellipsoid(center, rx, ry, rz, count=300):
        center = np.array(center)
        u = np.random.uniform(0, 2 * np.pi, count)
        v = np.random.uniform(0, np.pi, count)
        x = center[0] + rx * np.cos(u) * np.sin(v) + np.random.normal(0, 0.015, count)
        y = center[1] + ry * np.sin(u) * np.sin(v) + np.random.normal(0, 0.015, count)
        z = center[2] + rz * np.cos(v) + np.random.normal(0, 0.015, count)
        return np.column_stack([x, y, z])

    # 1. Wheels
    points.append(add_ring([-2.6, 0.0, 1.25], 1.25, 0.03, "xz", int(total_points * 0.12)))
    points.append(add_ring([2.6, 0.0, 1.25], 1.25, 0.03, "xz", int(total_points * 0.12)))

    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        spoke_end_r = [-2.6 + 1.25 * np.cos(angle), 0.0, 1.25 + 1.25 * np.sin(angle)]
        spoke_end_f = [2.6 + 1.25 * np.cos(angle), 0.0, 1.25 + 1.25 * np.sin(angle)]
        points.append(add_cylinder([-2.6, 0.0, 1.25], spoke_end_r, 0.015, 20))
        points.append(add_cylinder([2.6, 0.0, 1.25], spoke_end_f, 0.015, 20))

    # 2. Frame Tubes
    bb = [0.0, 0.0, 1.2]
    seat = [-1.1, 0.0, 2.7]
    head = [1.9, 0.0, 3.1]
    f_axle = [2.6, 0.0, 1.25]
    r_axle = [-2.6, 0.0, 1.25]

    points.append(add_cylinder(bb, seat, 0.05, int(total_points * 0.04)))
    points.append(add_cylinder(seat, head, 0.045, int(total_points * 0.05)))
    points.append(add_cylinder(bb, head, 0.05, int(total_points * 0.05)))
    points.append(add_cylinder(bb, r_axle, 0.035, int(total_points * 0.03)))
    points.append(add_cylinder(seat, r_axle, 0.035, int(total_points * 0.03)))
    points.append(add_cylinder(head, f_axle, 0.045, int(total_points * 0.03)))

    points.append(add_cylinder([1.9, 0.0, 3.1], [1.9, 0.0, 3.5], 0.035, 30))
    points.append(add_cylinder([1.9, -0.65, 3.5], [1.9, 0.65, 3.5], 0.035, 90))

    # 3. Rider Body
    hip = [-1.0, 0.0, 2.95]
    shoulder = [-0.1, 0.0, 4.15]
    head_c = [0.25, 0.0, 4.55]
    helmet_back = [0.05, 0.0, 4.65]

    points.append(add_cylinder(hip, shoulder, 0.16, int(total_points * 0.10)))
    points.append(add_ellipsoid(head_c, 0.22, 0.18, 0.20, int(total_points * 0.06)))
    points.append(add_ellipsoid(helmet_back, 0.18, 0.16, 0.15, int(total_points * 0.04)))

    elbow_l = [0.85, -0.55, 3.85]
    elbow_r = [0.85, 0.55, 3.85]
    points.append(add_cylinder(shoulder, elbow_l, 0.07, int(total_points * 0.03)))
    points.append(add_cylinder(shoulder, elbow_r, 0.07, int(total_points * 0.03)))
    points.append(add_cylinder(elbow_l, [1.9, -0.55, 3.45], 0.06, int(total_points * 0.03)))
    points.append(add_cylinder(elbow_r, [1.9, 0.55, 3.45], 0.06, int(total_points * 0.03)))

    knee_l = [-0.15, -0.42, 2.35]
    pedal_l = [0.0, -0.32, 0.75]
    knee_r = [-0.85, 0.42, 1.85]
    pedal_r = [0.0, 0.32, 1.65]

    points.append(add_cylinder(hip, knee_l, 0.10, int(total_points * 0.04)))
    points.append(add_cylinder(knee_l, pedal_l, 0.08, int(total_points * 0.03)))
    points.append(add_cylinder(hip, knee_r, 0.10, int(total_points * 0.04)))
    points.append(add_cylinder(knee_r, pedal_r, 0.08, int(total_points * 0.03)))

    return np.vstack(points)


# -----------------------------------------------------------------------------
# Sidebar Organized Controls
# -----------------------------------------------------------------------------
st.sidebar.header("Dataset Controls (Tabs 1 & 2)")
st.sidebar.caption("Modifying these alters the 3D statue dataset and recalculates PCA math and Scree plots.")

n_points_sel = st.sidebar.slider(
    "Sample Count / Point Density", 1000, 8000, 4000, step=1000, key="n_points_slider"
)

st.sidebar.markdown("---")
st.sidebar.header("Camera Controls (Tab 1 Only)")
st.sidebar.caption("Adjusts visual perspective in Tab 1 without altering mathematical variance.")

azimuth = st.sidebar.slider("Camera Azimuth (Horizontal)", 0, 360, 40, key="az_slider")
elevation = st.sidebar.slider("Camera Elevation (Vertical)", -90, 90, 15, key="el_slider")

# Generate dataset dynamically based on sidebar sample count
X_3d = generate_highres_cyclist_statue(n_points_sel)
X_mean = np.mean(X_3d, axis=0)
X_centered = X_3d - X_mean

pca_full = PCA(n_components=3)
pca_full.fit(X_centered)


# =============================================================================
# TAB 1 CONTENT
# =============================================================================
if selected_tab == tabs_list[0]:
    st.subheader("1. The 3D Object: Detailed Bronze Cyclist Statue")
    st.caption("A high-resolution 3D point-cloud representation of the cyclist statue in 3D space.")

    fig_3d = go.Figure()
    fig_3d.add_trace(
        go.Scatter3d(
            x=X_3d[:, 0],
            y=X_3d[:, 1],
            z=X_3d[:, 2],
            mode="markers",
            marker=dict(size=2, color=X_3d[:, 2], colorscale="earth", opacity=0.9),
            name="Statue Data Points",
        )
    )

    comp1 = pca_full.components_[0]
    comp2 = pca_full.components_[1]
    scale = 3.5
    for i, (comp, color, name) in enumerate(
        zip([comp1, comp2], ["#EF4444", "#F59E0B"], ["PC1 Vector", "PC2 Vector"])
    ):
        p_end = X_mean + comp * scale
        fig_3d.add_trace(
            go.Scatter3d(
                x=[X_mean[0], p_end[0]],
                y=[X_mean[1], p_end[1]],
                z=[X_mean[2], p_end[2]],
                mode="lines+text",
                line=dict(color=color, width=8),
                text=["", name],
                textposition="top center",
                name=name,
            )
        )

    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    cam_x = 1.8 * np.cos(el_rad) * np.cos(az_rad)
    cam_y = 1.8 * np.cos(el_rad) * np.sin(az_rad)
    cam_z = 1.8 * np.sin(el_rad)

    fig_3d.update_layout(
        scene=dict(
            xaxis_title="Length (X)",
            yaxis_title="Width (Y)",
            zaxis_title="Height (Z)",
            aspectmode="data",
            camera=dict(eye=dict(x=cam_x, y=cam_y, z=cam_z)),
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        height=480,
        showlegend=True,
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    def project_camera_view(data_3d, az_deg, el_deg):
        az_r = np.radians(az_deg)
        el_r = np.radians(el_deg)
        proj_u = np.array([-np.sin(az_r), np.cos(az_r), 0])
        proj_v = np.array(
            [
                -np.sin(el_r) * np.cos(az_r),
                -np.sin(el_r) * np.sin(az_r),
                np.cos(el_r),
            ]
        )
        u = np.dot(data_3d, proj_u)
        v = np.dot(data_3d, proj_v)
        var_captured = np.var(u) + np.var(v)
        total_var = np.sum(np.var(data_3d, axis=0))
        pct_var = min(100.0, (var_captured / total_var) * 100)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=u,
                y=v,
                mode="markers",
                marker=dict(size=2, color=data_3d[:, 2], colorscale="earth", opacity=0.85),
            )
        )
        fig.update_layout(
            xaxis=dict(showticklabels=False, title="Camera Width Axis"),
            yaxis=dict(showticklabels=False, title="Camera Height Axis"),
            margin=dict(l=10, r=10, b=10, t=10),
            height=380,
        )
        return fig, pct_var

    st.markdown("---")
    st.subheader("2. Comparing 2D Photographs from Different Camera Angles")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Bad Camera Angle (Bird's-Eye View)")
        fig_bad, var_bad = project_camera_view(X_centered, az_deg=0, el_deg=88)
        st.plotly_chart(fig_bad, use_container_width=True)
        st.metric("Information Preserved", f"{var_bad:.1f}%")
        st.warning("Information Lost: Features overlap and collapse.")

    with col2:
        st.markdown("### Optimal PCA Camera Angle")
        X_pca_2d = pca_full.transform(X_centered)[:, :2]
        fig_pca = go.Figure()
        fig_pca.add_trace(
            go.Scatter(
                x=X_pca_2d[:, 0],
                y=X_pca_2d[:, 1],
                mode="markers",
                marker=dict(size=2, color=X_3d[:, 2], colorscale="earth", opacity=0.85),
            )
        )
        pca_var_pct = np.sum(pca_full.explained_variance_ratio_[:2]) * 100
        fig_pca.update_layout(
            xaxis=dict(showticklabels=False, title="PC1 (Maximum Spread)"),
            yaxis=dict(showticklabels=False, title="PC2 (Perpendicular Spread)"),
            margin=dict(l=10, r=10, b=10, t=10),
            height=380,
        )
        st.plotly_chart(fig_pca, use_container_width=True)
        st.metric("Information Preserved", f"{pca_var_pct:.1f}%")
        st.success("Maximum Information Retained: Full structural detail captured.")


# =============================================================================
# TAB 2 CONTENT
# =============================================================================
elif selected_tab == tabs_list[1]:
    st.subheader("Mathematical Foundations: Variance & Linear Algebra")
    st.markdown(
        """
        Transitioning from our visual analogy to the underlying mathematics, PCA follows these formal steps:

        1. **Mean Centering:** Shift the data cloud so its center sits at the origin $(0, 0, 0)$:
           $$X_{\\text{centered}} = X - \\mu$$
        2. **Covariance Matrix ($C$):** Compute how features vary together across dimensions:
           $$C = \\frac{1}{n-1} X_{\\text{centered}}^T X_{\\text{centered}}$$
        3. **Eigen-decomposition:** Solve the characteristic equation to find the orthogonal axes (eigenvectors $v$) and amount of variance explained (eigenvalues $\\lambda$):
           $$C v = \\lambda v$$
        """
    )

    explained_var = pca_full.explained_variance_ratio_ * 100
    cumulative_var = np.cumsum(explained_var)

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("PC1 Information", f"{explained_var[0]:.1f}%")
    with col_m2:
        st.metric("PC2 Information", f"{explained_var[1]:.1f}%")
    with col_m3:
        st.metric("PC3 Information", f"{explained_var[2]:.1f}%")

    fig_scree = go.Figure()

    fig_scree.add_trace(
        go.Bar(
            x=["PC1", "PC2", "PC3"],
            y=explained_var,
            name="Individual Variance (%)",
            marker_color="#2563EB",
        )
    )

    fig_scree.add_trace(
        go.Scatter(
            x=["PC1", "PC2", "PC3"],
            y=cumulative_var,
            name="Cumulative Variance (%)",
            mode="lines+markers",
            line=dict(color="#EF4444", width=4),
            marker=dict(size=10),
        )
    )

    target_thresh = st.slider("Target Information Threshold (%)", 50, 100, 90, step=5, key="target_thresh_slider")
    fig_scree.add_hline(
        y=target_thresh,
        line_dash="dash",
        line_color="#10B981",
        annotation_text=f"Target: {target_thresh}% Variance",
        annotation_position="bottom right",
    )

    fig_scree.update_layout(
        title=f"Scree Plot for Current Dataset (Sample Count = {n_points_sel:,})",
        yaxis_title="Percentage of Total Variance (%)",
        yaxis=dict(range=[0, 105]),
        height=420,
        margin=dict(l=20, r=20, b=20, t=40),
    )

    st.plotly_chart(fig_scree, use_container_width=True)

    st.info(
        f"Notes for Tab 2:\n"
        f"- Dataset Resolution: Changing the sidebar sample count ({n_points_sel:,} points) recalculates the covariance matrix and eigen-decomposition live.\n"
        f"- PC1 ({explained_var[0]:.1f}%): Corresponds to the largest eigenvalue. It points directly along the longest axis of the data spread.\n"
        f"- PC2 ({explained_var[1]:.1f}%): Corresponds to the second largest eigenvalue, orthogonal (perpendicular) to PC1.\n"
        f"- Cumulative Total ({cumulative_var[1]:.1f}%): Retaining just the top two principal components preserves {cumulative_var[1]:.1f}% of total variance."
    )


# =============================================================================
# TAB 3 CONTENT
# =============================================================================
elif selected_tab == tabs_list[2]:
    st.subheader("Real World Applications: High-Dimensional Dataset Reduction")
    st.markdown(
        """
        In practice, real-world machine learning datasets have dozens or hundreds of features.
        PCA reduces these high-dimensional spaces down to 2D so we can visualize hidden clusters and patterns.
        """
    )

    dataset_choice = st.selectbox(
        "Select Benchmark Dataset:",
        ["Wine Cultivars (13 Dimensions)", "Iris Flowers (4 Dimensions)"],
        index=0,
        key="dataset_selectbox",
    )

    if "Wine" in dataset_choice:
        raw_data = load_wine()
        df_x = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
        labels = [raw_data.target_names[i] for i in raw_data.target]
        title_name = "Wine Dataset (13 Features -> 2D Projection)"
    else:
        raw_data = load_iris()
        df_x = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
        labels = [raw_data.target_names[i] for i in raw_data.target]
        title_name = "Iris Dataset (4 Features -> 2D Projection)"

    # Standardize data prior to PCA
    X_std = (df_x - df_x.mean()) / df_x.std()

    # Fit PCA
    pca_real = PCA(n_components=2)
    X_real_2d = pca_real.fit_transform(X_std)

    var_real_pc1 = pca_real.explained_variance_ratio_[0] * 100
    var_real_pc2 = pca_real.explained_variance_ratio_[1] * 100
    total_real_var = var_real_pc1 + var_real_pc2

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("### 1. 2D Cluster Projection")
        st.caption(f"Captured **{total_real_var:.1f}%** of total dataset variance using only 2 components.")

        df_proj = pd.DataFrame(
            X_real_2d, columns=["Principal Component 1", "Principal Component 2"]
        )
        df_proj["Class"] = labels

        fig_clusters = px.scatter(
            df_proj,
            x="Principal Component 1",
            y="Principal Component 2",
            color="Class",
            title=title_name,
            symbol="Class",
            opacity=0.85,
        )
        fig_clusters.update_traces(marker=dict(size=9))
        fig_clusters.update_layout(height=450, margin=dict(l=10, r=10, b=10, t=40))
        st.plotly_chart(fig_clusters, use_container_width=True)

    with col_r2:
        st.markdown("### 2. Feature Loadings (Biplot Weights)")
        st.caption("Shows which original features contribute most strongly to PC1 and PC2.")

        loadings = pd.DataFrame(
            pca_real.components_.T,
            columns=["PC1 Loading", "PC2 Loading"],
            index=df_x.columns,
        )

        fig_loadings = px.bar(
            loadings,
            barmode="group",
            title="Feature Contributions to Principal Axes",
        )
        fig_loadings.update_layout(
            height=450,
            margin=dict(l=10, r=10, b=10, t=40),
            xaxis_title="Original Features",
            yaxis_title="Loading Weight",
        )
        st.plotly_chart(fig_loadings, use_container_width=True)

    st.info(
        f"Notes for Tab 3:\n"
        f"- Dimension Reduction: We took {df_x.shape[1]} original features and flattened them into 2 Principal Components while retaining {total_real_var:.1f}% of all original information.\n"
        f"- Cluster Separability: Notice how distinct classes naturally separate into distinct clusters in 2D space without needing all original features.\n"
        f"- Feature Loadings & Direction: The sign (+/-) indicates direction, NOT a reduction in variance. A loading of -0.8 contributes just as much variance/information to the component as +0.8, but in the opposite direction along that axis.\n"
        f"- Negative Loadings: Indicate an inverse relationship with the principal component axis (e.g., as that feature increases, the sample's score along that component decreases)."
    )


# =============================================================================
# TAB 4 CONTENT: HIGH-RESOLUTION IMAGE COMPRESSION & RECONSTRUCTION
# =============================================================================
elif selected_tab == tabs_list[3]:
    st.subheader("High-Resolution Image Compression & Reconstruction via PCA")
    st.markdown(
        """
        PCA can compress images by treating each pixel position as a feature dimension.
        By projecting an 8×8 image (64 dimensions) onto a smaller subset of $k$ principal components,
        we can reconstruct the image while discarding noise and high-frequency details.
        
        *Note: Images are smoothly upsampled to 128×128 resolution using bicubic interpolation for visual clarity while preserving exact mathematical values.*
        """
    )

    digits = load_digits()
    X_digits = digits.data
    y_digits = digits.target
    images_digits = digits.images

    col_ctrl1, col_ctrl2 = st.columns([1, 2])

    with col_ctrl1:
        sample_idx = st.slider("Select Handwritten Digit Sample Index:", 0, len(X_digits) - 1, 0, key="digit_sample_slider")
        target_digit = y_digits[sample_idx]
        st.caption(f"Selected Image Ground Truth Label: **Digit '{target_digit}'**")

    with col_ctrl2:
        k_components = st.slider("Number of Retained Principal Components (k):", 1, 64, 10, key="k_components_slider")

    # Fit PCA across the digits dataset
    pca_digits = PCA(n_components=64)
    pca_digits.fit(X_digits)

    # Project selected sample down to k components and reconstruct
    X_sample = X_digits[sample_idx : sample_idx + 1]
    X_sample_pca = pca_digits.transform(X_sample)[:, :k_components]

    # Reconstruction formula: X_reconstructed = X_pca * Components + Mean
    X_reconstructed = (
        np.dot(X_sample_pca, pca_digits.components_[:k_components, :])
        + pca_digits.mean_
    )
    img_orig = images_digits[sample_idx]
    img_recon = X_reconstructed.reshape(8, 8)

    # Smooth Upsampling (8x8 -> 128x128) via Scipy Bicubic Interpolation
    scale_factor = 16
    img_orig_smooth = zoom(img_orig, scale_factor, order=3)
    img_recon_smooth = zoom(img_recon, scale_factor, order=3)

    cum_var_digits = np.sum(pca_digits.explained_variance_ratio_[:k_components]) * 100
    compression_ratio = 64 / k_components

    st.markdown("---")
    col_img1, col_img2, col_img3 = st.columns([1, 1, 1])

    with col_img1:
        st.markdown("### 1. Original Image")
        st.caption("Full 64 Dimensions (8×8 Pixels, Upsampled)")
        fig_orig = px.imshow(
            img_orig_smooth,
            color_continuous_scale="gray",
            labels=dict(x="X Axis", y="Y Axis"),
            origin="upper",
        )
        fig_orig.update_xaxes(showticklabels=False)
        fig_orig.update_yaxes(showticklabels=False)
        fig_orig.update_layout(height=300, margin=dict(l=10, r=10, b=10, t=30))
        st.plotly_chart(fig_orig, use_container_width=True)

    with col_img2:
        st.markdown(f"### 2. Reconstructed Image (k={k_components})")
        st.caption(f"Reconstructed using top {k_components} Principal Components")
        fig_recon = px.imshow(
            img_recon_smooth,
            color_continuous_scale="gray",
            labels=dict(x="X Axis", y="Y Axis"),
            origin="upper",
        )
        fig_recon.update_xaxes(showticklabels=False)
        fig_recon.update_yaxes(showticklabels=False)
        fig_recon.update_layout(height=300, margin=dict(l=10, r=10, b=10, t=30))
        st.plotly_chart(fig_recon, use_container_width=True)

    with col_img3:
        st.markdown("### 3. Compression Metrics")
        st.metric("Retained Variance", f"{cum_var_digits:.1f}%")
        st.metric("Compression Ratio", f"{compression_ratio:.1f}x reduction")
        if k_components <= 5:
            st.warning("Low component count: Broad digit structural shape visible.")
        elif k_components <= 20:
            st.success("Optimal compression: Digit clearly recognizable with >85% variance!")
        else:
            st.info("High component count: Captures fine-grained pixel details.")

    st.markdown("---")
    st.markdown("### 4. Eigen-Digits (Top Principal Component Basis Vectors)")
    st.caption("These are the fundamental smooth 'building blocks' (eigenvectors) used to construct every digit image in the dataset.")

    eigen_cols = st.columns(6)
    for idx, col in enumerate(eigen_cols):
        eigen_digit = pca_digits.components_[idx].reshape(8, 8)
        eigen_digit_smooth = zoom(eigen_digit, scale_factor, order=3)
        fig_eigen = px.imshow(
            eigen_digit_smooth,
            color_continuous_scale="RdBu",
            title=f"PC{idx+1}",
            origin="upper",
        )
        fig_eigen.update_layout(
            height=180,
            margin=dict(l=5, r=5, b=5, t=30),
            coloraxis_showscale=False
        )
        fig_eigen.update_xaxes(showticklabels=False)
        fig_eigen.update_yaxes(showticklabels=False)
        col.plotly_chart(fig_eigen, use_container_width=True)

    st.info(
        f"Notes for Tab 4:\n"
        f"- High-Resolution Visualization: Applying bicubic upsampling smoothly renders pixel values without altering underlying PCA mathematical projections.\n"
        f"- Feature Representation: Each 8×8 image starts with 64 pixel dimensions. PCA finds orthogonal eigen-digit basis patterns.\n"
        f"- Lossy Compression: With k={k_components} components out of 64, we retain {cum_var_digits:.1f}% of total image variance while achieving a {compression_ratio:.1f}x compression ratio.\n"
        f"- Practical Application: Reduces feature space, accelerates training, and removes noise in machine learning pipelines."
    )