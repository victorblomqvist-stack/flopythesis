# Here the tensors will be plotted to evaluate form

# Input
    # Conductivity tensor for each rotation, stored in a csv file. 
# Output
     # Shape pof conductivity tensor in 3D. Plotted to evaluate if it is ellipse-like.

"""Track rotated permeability vectors and test for ellipse-like behavior.

Expected CSV columns:
angle_deg,k_xx,k_xy,k_xz,k_yx,k_yy,k_yz,k_zx,k_zy,k_zz
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import pyvista as pv
except ImportError:  # pragma: no cover
    pv = None


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save a figure so results are visible even if GUI windows do not appear."""
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved figure: {output_path}")


def get_project_root() -> Path:
    """Return the directory containing this script and the CSV."""
    return Path(__file__).resolve().parent


def read_tensor_csv(csv_path: Path) -> pd.DataFrame:
    """Load tensor data and enforce numeric dtype for all columns."""
    df = pd.read_csv(csv_path)
    numeric_cols = [
        "angle_deg",
        "k_xx",
        "k_xy",
        "k_xz",
        "k_yx",
        "k_yy",
        "k_yz",
        "k_zx",
        "k_zy",
        "k_zz",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=numeric_cols).sort_values("angle_deg").reset_index(drop=True)


def build_tensor(row: pd.Series) -> np.ndarray:
    """Build 3x3 permeability tensor matrix from one dataframe row."""
    return np.array(
        [
            [row["k_xx"], row["k_xy"], row["k_xz"]],
            [row["k_yx"], row["k_yy"], row["k_yz"]],
            [row["k_zx"], row["k_zy"], row["k_zz"]],
        ],
        dtype=float,
    )


def enforce_symmetry(tensor: np.ndarray) -> np.ndarray:
    """Average tensor with its transpose to avoid tiny asymmetries."""
    return 0.5 * (tensor + tensor.T)


def principal_vectors_from_tensors(df: pd.DataFrame) -> np.ndarray:
    """Return one tracked vector per rotation: principal axis scaled by sqrt(lambda_max)."""
    vectors = []
    for _, row in df.iterrows():
        tensor = enforce_symmetry(build_tensor(row))
        eigvals, eigvecs = np.linalg.eigh(tensor)
        idx = int(np.argmax(eigvals))
        vmax = eigvecs[:, idx]
        scale = float(np.sqrt(max(eigvals[idx], 0.0)))
        vectors.append(vmax * scale)
    return np.asarray(vectors)


def ellipse_fit_residual(points_2d: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit a general conic to 2D points and return RMS algebraic residual."""
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    design = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
    _, _, vh = np.linalg.svd(design, full_matrices=False)
    coeff = vh[-1, :]
    residual = design @ coeff
    rms = float(np.sqrt(np.mean(residual**2)))
    return rms, coeff


def classify_conic(coeff: np.ndarray) -> str:
    """Classify fitted conic using discriminant B^2 - 4AC."""
    a, b, c, _, _, _ = coeff
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return "ellipse-like"
    if np.isclose(disc, 0.0):
        return "parabola-like"
    return "hyperbola-like"


def project_to_best_plane(points_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D points to their best-fit 2D plane using PCA."""
    center = points_3d.mean(axis=0)
    shifted = points_3d - center
    _, _, vh = np.linalg.svd(shifted, full_matrices=False)
    basis = vh[:2, :]  # first 2 principal directions
    points_2d = shifted @ basis.T
    return points_2d, center, basis


def plot_rotation_vectors(points_3d: np.ndarray, angles: np.ndarray) -> None:
    """Plot each rotation vector from origin and tip trajectory in 3D."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    centroid = points_3d.mean(axis=0)

    colors = plt.cm.viridis(np.linspace(0, 1, len(angles)))
    for i, vec in enumerate(points_3d):
        ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]],
                color=colors[i], linewidth=1.1)
        ax.scatter(vec[0], vec[1], vec[2], color=colors[i], s=24)

    # Connect vector tips to show the tracked trajectory through rotation.
    ax.plot(points_3d[:, 0], points_3d[:, 1],
            points_3d[:, 2], "k--", linewidth=1.5, alpha=0.6)

    # Plot true tensor origin and point-cloud centroid as separate references.
    ax.scatter(0.0, 0.0, 0.0, marker="o", s=70,
               color="black", label="origin (0,0,0)")
    ax.scatter(centroid[0], centroid[1], centroid[2], marker="*", s=220,
               color="crimson", edgecolors="black", label="tip centroid")
    ax.legend(loc="upper left")

    max_extent = float(np.max(np.abs(points_3d)))
    if max_extent > 0:
        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-max_extent, max_extent)
        ax.set_zlim(-max_extent, max_extent)

    ax.set_title("Tracked Vector for Each Rotation Angle")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    save_figure(fig, get_project_root() / "tracked_vectors_3d.png")
    plt.show()


def plot_2d_projections(points_3d: np.ndarray, angles: np.ndarray) -> None:
    """Plot xy, xz, yz projections to visually inspect ellipse-like traces."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    centroid = points_3d.mean(axis=0)
    projections = [
        (0, 1, "XY"),
        (0, 2, "XZ"),
        (1, 2, "YZ"),
    ]

    for ax, (i, j, label) in zip(axes, projections):
        ax.scatter(points_3d[:, i], points_3d[:, j],
                   c=angles, cmap="viridis", s=45)
        ax.plot(points_3d[:, i], points_3d[:, j], "k--", alpha=0.55)
        ax.scatter(0.0, 0.0, marker="o", s=55,
                   color="black")
        ax.scatter(centroid[i], centroid[j], marker="*", s=180,
                   color="crimson", edgecolors="black")
        for idx, ang in enumerate(angles):
            ax.text(points_3d[idx, i], points_3d[idx, j],
                    f"{int(ang)}", fontsize=7)
        ax.set_title(f"{label} projection")
        ax.set_xlabel(label[0].lower())
        ax.set_ylabel(label[1].lower())
        ax.axis("equal")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    save_figure(fig, get_project_root() / "tracked_vectors_2d_projections.png")
    plt.show()


def evaluate_ellipse_likeness(points_3d: np.ndarray) -> None:
    """Project tracked points to best-fit plane and evaluate conic type."""
    points_2d, _, _ = project_to_best_plane(points_3d)
    rms, coeff = ellipse_fit_residual(points_2d)
    shape_type = classify_conic(coeff)

    x = points_2d[:, 0]
    y = points_2d[:, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=50)
    ax.plot(x, y, "k--", alpha=0.6)
    ax.scatter(0.0, 0.0, marker="*", s=200,
               color="crimson", edgecolors="black")
    for i in range(len(x)):
        ax.text(x[i], y[i], str(i), fontsize=8)
    ax.set_title("Tracked points projected to best-fit plane")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, get_project_root() / "ellipse_check_best_plane.png")
    plt.show()

    print("\nEllipse-likeness check (data-driven, not assumed):")
    print(f"  Conic classification: {shape_type}")
    print(f"  RMS algebraic residual: {rms:.3e}")
    print("  Lower residual and ellipse-like classification support an elliptical trend.")


def build_nodes_and_faces(points_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build node and triangular face connectivity from tracked 3D points."""
    if pv is None:
        raise ImportError(
            "PyVista is not installed. Install with: pip install pyvista")

    if len(points_3d) < 3:
        raise ValueError(
            "At least 3 points are required to build a surface mesh.")

    # Remove near-duplicate points before triangulation.
    scale = max(float(np.ptp(points_3d, axis=0).max()), 1.0)
    tol = 1e-10 * scale
    unique_nodes = [points_3d[0]]
    for p in points_3d[1:]:
        if min(np.linalg.norm(p - q) for q in unique_nodes) > tol:
            unique_nodes.append(p)
    nodes_in = np.asarray(unique_nodes, dtype=float)

    if len(nodes_in) < 3:
        raise ValueError(
            "Not enough unique points to construct triangular faces.")

    cloud = pv.PolyData(nodes_in)
    surface = None

    # Prefer outer shell in 3D: tetrahedralize then keep only external surface.
    try:
        volume = cloud.delaunay_3d()
        surface = volume.extract_surface(
            algorithm="dataset_surface").triangulate()
        if surface.n_cells == 0:
            surface = None
    except Exception:
        surface = None

    # Fallback for near-planar data.
    if surface is None:
        surface = cloud.delaunay_2d().triangulate()

    if surface.n_cells == 0:
        raise ValueError(
            "Could not construct an outer surface mesh from provided points.")

    nodes = np.asarray(surface.points)
    faces_flat = np.asarray(surface.faces)
    faces = faces_flat.reshape(-1, 4)[:, 1:4]
    return nodes, faces, faces_flat


def plot_pyvista_nodes_and_faces(
    nodes: np.ndarray,
    faces_flat: np.ndarray,
    output_path: Path,
) -> None:
    """Plot nodes and face mesh in 3D with PyVista and save screenshot."""
    if pv is None:
        raise ImportError(
            "PyVista is not installed. Install with: pip install pyvista")

    mesh = pv.PolyData(nodes, faces_flat)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _add_scene(plotter: "pv.Plotter") -> None:
        plotter.add_mesh(mesh, color="lightsteelblue",
                         opacity=0.45, show_edges=True)
        plotter.add_points(
            nodes,
            color="crimson",
            point_size=12,
            render_points_as_spheres=True,
        )
        plotter.add_axes()
        plotter.add_title("Tensor Vector Mesh (Nodes + Faces)")

    try:
        # Show interactive window and save screenshot in one call.
        plotter = pv.Plotter(window_size=(1200, 800), off_screen=False)
        _add_scene(plotter)
        plotter.show(screenshot=str(output_path))
    except Exception:
        # Fallback for environments where interactive rendering is unavailable.
        plotter = pv.Plotter(window_size=(1200, 800), off_screen=True)
        _add_scene(plotter)
        plotter.screenshot(str(output_path))
        plotter.close()
    print(f"Saved figure: {output_path}")


def create_and_plot_tensor_mesh(points_3d: np.ndarray) -> None:
    """Create node/face lists and visualize resulting mesh in 3D."""
    nodes, faces, faces_flat = build_nodes_and_faces(points_3d)
    print("\nPyVista mesh summary:")
    print(f"  Number of nodes: {len(nodes)}")
    print(f"  Number of triangular faces: {len(faces)}")
    print("  First 5 nodes:\n", nodes[:5])
    print("  First 5 faces (node indices):\n", faces[:5])

    output_path = get_project_root() / "tracked_vectors_pyvista_mesh.png"
    plot_pyvista_nodes_and_faces(nodes, faces_flat, output_path)


def main() -> None:
    root = get_project_root()
    # preferred_csv = root / "permeability_tensors_3d_rot_0_360.csv"
    # fallback_csv = root / "permeability_tensors_3d_rot_0_180.csv"

    preferred_csv = root / "tensor_sphere_rotations_10deg.csv"
    fallback_csv = root / "tensor_sphere_rotations_10deg.csv"

    if preferred_csv.exists():
        csv_path = preferred_csv
    elif fallback_csv.exists():
        csv_path = fallback_csv
    else:
        raise FileNotFoundError(
            f"CSV not found. Expected one of: {preferred_csv} or {fallback_csv}"
        )

    df = read_tensor_csv(csv_path)
    angles = df["angle_deg"].to_numpy()
    tracked_vectors = principal_vectors_from_tensors(df)

    print("Using CSV:", csv_path.name)
    print("Loaded rows:", len(df))
    print("Angles:", angles)

    plot_rotation_vectors(tracked_vectors, angles)
    plot_2d_projections(tracked_vectors, angles)
    evaluate_ellipse_likeness(tracked_vectors)
    create_and_plot_tensor_mesh(tracked_vectors)


if __name__ == "__main__":
    main()
