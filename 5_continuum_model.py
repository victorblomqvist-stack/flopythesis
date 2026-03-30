# Input: Conductivity tensor for each rotation, stored in a csv file.
# Output: Continuum model with anisotropic conductivity, visualized in 3D with PyVista.

"""Build a continuum MODFLOW 6 model where hydraulic conductivity comes from a tensor.

Workflow:
1) Read permeability tensor(s) from CSV (m^2).
2) Select one tensor (or average all tensors).
3) Convert permeability tensor to hydraulic conductivity tensor (m/s).
4) Use principal values and horizontal orientation in a FloPy continuum model.
"""

from pathlib import Path
import os
import flopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from flopy.export.vtk import Vtk
from flopy.utils.gridgen import Gridgen
from shapely.geometry import Polygon


# ----------------------- User settings -----------------------
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "permeability_tensors_3d_rot_0_360.csv"

# Choose how to pick tensor data: "angle" or "average"
TENSOR_SELECTION = "angle"
TARGET_ANGLE_DEG = 45.0

# Convert intrinsic permeability (m^2) -> hydraulic conductivity (m/s)
RHO = 1000.0      # kg/m^3
G = 9.81          # m/s^2
MU = 1.0e-3       # Pa*s

# Model and discretization
MODEL_NAME = "cube"
SIM_WS = ROOT / "continuum_ws"

# Match provided reference model parameters.
H_LEFT = 10.0
H_RIGHT = 1.0
H2 = 10.0
NLAY = 10
R = 10   # rows
C = 10   # columns
X_LEN = 100.0
Y_LEN = 100.0
Z_LEN = 100.0

# Use full domain length over number of cells so extent is exactly 100 m.
delr = X_LEN / C
delc = Y_LEN / R
TOP = 0.0
BOTM = np.linspace(-Z_LEN / NLAY, -Z_LEN, NLAY)

# Set to True only if mf6 executable is available in PATH or with exe_name path.
RUN_MODEL = True

# PyVista plotting options
PLOT_WITH_PYVISTA = True
SHOW_PYVISTA_INTERACTIVE = True
SHOW_PYVISTA_HEAD_ONLY_INTERACTIVE = True

# Optional 2D layer map (this is where "layer 9" came from).
PLOT_2D_LAYER_MAP = False

# Gridgen refinement options
USE_GRIDGEN_REFINEMENT = True
GRIDGEN_EXE = "gridgen_x64"


def read_tensor_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Tensor CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cols = [
        "angle_deg", "k_xx", "k_xy", "k_xz",
        "k_yx", "k_yy", "k_yz", "k_zx", "k_zy", "k_zz",
    ]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols).sort_values("angle_deg").reset_index(drop=True)
    return df


def row_to_tensor(row: pd.Series) -> np.ndarray:
    t = np.array(
        [
            [row["k_xx"], row["k_xy"], row["k_xz"]],
            [row["k_yx"], row["k_yy"], row["k_yz"]],
            [row["k_zx"], row["k_zy"], row["k_zz"]],
        ],
        dtype=float,
    )
    # Enforce symmetry to avoid numerical asymmetry from data noise.
    return 0.5 * (t + t.T)


def select_permeability_tensor(df: pd.DataFrame) -> tuple[np.ndarray, float]:
    if TENSOR_SELECTION == "average":
        tensors = [row_to_tensor(r) for _, r in df.iterrows()]
        return np.mean(tensors, axis=0), np.nan

    # Default: pick row closest to target angle.
    idx = int(np.argmin(np.abs(df["angle_deg"].to_numpy() - TARGET_ANGLE_DEG)))
    row = df.iloc[idx]
    return row_to_tensor(row), float(row["angle_deg"])


def permeability_to_hydraulic_conductivity(k_perm: np.ndarray) -> np.ndarray:
    factor = (RHO * G) / MU
    return factor * k_perm


def principal_k_and_angle(k_tensor: np.ndarray) -> tuple[float, float, float, float]:
    """Get principal K values and major-axis azimuth angle in xy plane.

    Returns:
        k11, k22, k33, angle1_deg
    """
    eigvals, eigvecs = np.linalg.eigh(k_tensor)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Clip tiny negatives that can appear from numeric noise.
    eigvals = np.maximum(eigvals, 1e-30)

    v_major = eigvecs[:, 0]
    angle1_deg = float(np.degrees(np.arctan2(v_major[1], v_major[0])))

    k11, k22, k33 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
    return k11, k22, k33, angle1_deg


def build_refined_gridprops_disv() -> dict:
    """Build DISV gridprops from Gridgen using a central polygon refinement."""
    sim_base = flopy.mf6.MFSimulation(sim_name="base", sim_ws=str(SIM_WS))
    gwf_base = flopy.mf6.ModflowGwf(sim_base, modelname="base")
    flopy.mf6.ModflowGwfdis(
        gwf_base,
        nlay=NLAY,
        nrow=R,
        ncol=C,
        delr=delr,
        delc=delc,
        top=TOP,
        botm=BOTM,
    )

    gridgen_ws = ROOT / MODEL_NAME
    gridgen_ws.mkdir(parents=True, exist_ok=True)

    g = Gridgen(gwf_base.modelgrid, model_ws=str(
        gridgen_ws), exe_name=GRIDGEN_EXE)

    center_poly = [Polygon([(40, 40), (60, 40), (60, 60), (40, 60)])]
    g.add_refinement_features(center_poly, "polygon", 3, list(range(NLAY)))
    g.build()

    vtk_path = os.path.join(str(gridgen_ws), "qtg_sv.vtu")
    if os.path.exists(vtk_path):
        refined_grid = pv.read(vtk_path)
        out_png = ROOT / "gridgen_refined_grid.png"
        p = pv.Plotter(off_screen=True)
        p.add_mesh(refined_grid, show_edges=True, opacity=1)
        p.add_axes()
        p.show(screenshot=str(out_png), title="3D Refined Grid from Gridgen")
        print(f"Saved Gridgen refined-grid plot: {out_png}")
    else:
        print(f"VTK file not found: {vtk_path}")

    gridprops = g.get_gridprops_disv()
    return gridprops


def build_continuum_model(k11: float, k22: float, k33: float, angle1_deg: float) -> flopy.mf6.MFSimulation:
    SIM_WS.mkdir(parents=True, exist_ok=True)

    sim = flopy.mf6.MFSimulation(
        sim_name=MODEL_NAME, sim_ws=str(SIM_WS), exe_name="mf6")
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=1,
                          perioddata=[(1.0, 1, 1.0)])
    flopy.mf6.ModflowIms(sim, complexity="SIMPLE")

    gwf = flopy.mf6.ModflowGwf(sim, modelname=MODEL_NAME, save_flows=True)

    if USE_GRIDGEN_REFINEMENT:
        gridprops = build_refined_gridprops_disv()
        flopy.mf6.ModflowGwfdisv(gwf, pname="disv", **gridprops)
        ncpl = int(gridprops["ncpl"])
    else:
        flopy.mf6.ModflowGwfdis(
            gwf,
            nlay=NLAY,
            nrow=R,
            ncol=C,
            delr=delr,
            delc=delc,
            top=TOP,
            botm=BOTM,
        )
        ncpl = R * C

    # Initial condition matched to provided parameter h2.
    flopy.mf6.ModflowGwfic(gwf, strt=H2)

    # Use principal conductivity ratios and horizontal rotation angle from the tensor.
    # k = k11, k22 = ratio*k, k33 = ratio*k.
    k22_ratio = k22 / k11
    k33_ratio = k33 / k11
    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,
        k=k11,
        k22=k22_ratio,
        k33=k33_ratio,
        angle1=angle1_deg,
        save_specific_discharge=True,
    )

    chd_spd = []
    if USE_GRIDGEN_REFINEMENT:
        mg = gwf.modelgrid
        xc = np.array(mg.xcellcenters).flatten()

        for icell in range(ncpl):
            on_left = xc[icell] < delr
            on_right = xc[icell] > X_LEN - delr
            if on_left or on_right:
                for lay in range(NLAY):
                    if on_left:
                        chd_spd.append(((lay, icell), H_LEFT))
                    if on_right:
                        chd_spd.append(((lay, icell), H_RIGHT))
    else:
        for lay in range(NLAY):
            for row in range(R):
                chd_spd.append(((lay, row, 0), H_LEFT))
                chd_spd.append(((lay, row, C - 1), H_RIGHT))

    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd)
    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=[f"{MODEL_NAME}.cbb"],
        head_filerecord=[f"{MODEL_NAME}.hds"],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    return sim


def quick_plot_last_layer_head(sim: flopy.mf6.MFSimulation) -> None:
    gwf = sim.get_model(MODEL_NAME)
    head_path = SIM_WS / f"{MODEL_NAME}.hds"
    hds = flopy.utils.HeadFile(str(head_path))
    head = hds.get_data(kstpkper=(0, 0))

    # For DISV, head is typically (nlay, ncpl). For DIS, it is (nlay, nrow, ncol).
    # PlotMapView expects the full array and a valid layer index.
    nlay_head = head.shape[0] if head.ndim > 1 else 1
    layer_to_plot = min(NLAY - 1, nlay_head - 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    pmv = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=layer_to_plot)
    quad = pmv.plot_array(head, cmap="viridis")
    pmv.plot_grid(alpha=0.15)
    ax.set_title(f"Head distribution (layer {layer_to_plot})")
    fig.colorbar(quad, ax=ax, label="Head [m]")
    fig.tight_layout()
    out_png = ROOT / "continuum_head_bottom_layer.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {out_png}")
    plt.show()


def _tensor_overlay_meshes(k_tensor: np.ndarray, center: tuple[float, float, float]) -> pv.PolyData:
    """Create a tensor ellipsoid in model coordinates with scalar coloring."""
    eigvals, eigvecs = np.linalg.eigh(k_tensor)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1e-30)
    eigvecs = eigvecs[:, order]

    # Scale ellipsoid to model size while preserving tensor anisotropy ratios.
    rel_radii = np.sqrt(eigvals / eigvals.max())
    max_radius = 0.15 * min(X_LEN, Y_LEN, Z_LEN)
    radii = rel_radii * max_radius

    ellipsoid = pv.ParametricEllipsoid(radii[0], radii[1], radii[2],
                                       u_res=48, v_res=48, w_res=24)
    pts = ellipsoid.points @ eigvecs.T
    pts = pts + np.asarray(center)
    ellipsoid.points = pts

    # Color tensor surface by directional conductivity n^T K n.
    centered_pts = ellipsoid.points - np.asarray(center)
    norms = np.linalg.norm(centered_pts, axis=1)
    norms[norms == 0.0] = 1.0
    nvec = centered_pts / norms[:, None]
    tensor_value = np.einsum("ni,ij,nj->n", nvec, k_tensor, nvec)
    ellipsoid["tensor_value"] = tensor_value

    return ellipsoid


def plot_continuum_with_pyvista(sim: flopy.mf6.MFSimulation, k_tensor: np.ndarray, angle1_deg: float) -> None:
    """Visualize the continuum model grid in 3D with PyVista.

    If a head file exists (after running MF6), it is used as scalar coloring.
    Otherwise, the model is colored by layer index.
    """
    gwf = sim.get_model(MODEL_NAME)

    vtk = Vtk(model=gwf, binary=False, smooth=False)
    vtk.add_model(gwf)
    grid = vtk.to_pyvista()

    head_path = SIM_WS / f"{MODEL_NAME}.hds"
    if head_path.exists():
        hds = flopy.utils.HeadFile(str(head_path))
        head = hds.get_data(kstpkper=(0, 0)).flatten()
        grid["head"] = head
        scalar_name = "head"
        cmap = "viridis"
    else:
        ncpl = int(getattr(gwf.modelgrid, "ncpl", R * C))
        layer_id = np.repeat(np.arange(NLAY), ncpl)
        grid["layer_id"] = layer_id
        scalar_name = "layer_id"
        cmap = "plasma"

    # Tensor-only view in the same model coordinate system.
    center = grid.center
    tensor_ellipsoid = _tensor_overlay_meshes(k_tensor, center)

    # Create principal axis vectors for tensor visualization.
    eigvals, eigvecs = np.linalg.eigh(k_tensor)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1e-30)
    eigvecs = eigvecs[:, order]

    # Scale arrows proportional to principal values.
    arrow_scale = 0.15 * min(X_LEN, Y_LEN, Z_LEN)
    colors = ["red", "green", "blue"]
    arrows = []
    for i, (eigenval, eigenvec, color) in enumerate(zip(eigvals, eigvecs.T, colors)):
        arrow_length = eigenvec * arrow_scale * \
            np.sqrt(eigenval / eigvals.max())
        arrow_end = center + arrow_length
        arrow = pv.Arrow(start=center, direction=arrow_length, scale=1)
        arrows.append((arrow, color))

    # Always save an image, so you can inspect results even without GUI support.
    out_png = ROOT / "continuum_model_pyvista.png"
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(tensor_ellipsoid, scalars="tensor_value", cmap="viridis",
                     opacity=0.95, show_edges=False)
    for arrow, color in arrows:
        plotter.add_mesh(arrow, color=color, opacity=0.8)
    plotter.add_scalar_bar(title="tensor_value")
    plotter.add_axes()
    plotter.show_grid()
    plotter.camera_position = "iso"
    plotter.show(screenshot=str(out_png))
    print(f"Saved PyVista plot: {out_png}")

    if SHOW_PYVISTA_INTERACTIVE:
        plotter2 = pv.Plotter()
        plotter2.add_mesh(tensor_ellipsoid, scalars="tensor_value", cmap="viridis",
                          opacity=0.95, show_edges=False)
        for arrow, color in arrows:
            plotter2.add_mesh(arrow, color=color, opacity=0.8)
        plotter2.add_scalar_bar(title="tensor_value")
        plotter2.add_axes()
        plotter2.show_grid()
        plotter2.camera_position = "iso"
        plotter2.show(title="Tensor only")

    # Separate head-only plot: full 3D cube with every cell colored by head.
    head_only_png = ROOT / "continuum_model_head_only_pyvista.png"

    plotter_head = pv.Plotter(off_screen=True)
    plotter_head.add_mesh(grid, scalars=scalar_name, cmap=cmap,
                          show_edges=True, opacity=1.0)
    plotter_head.add_scalar_bar(title=scalar_name)
    plotter_head.add_axes()
    plotter_head.show_grid()
    plotter_head.camera_position = "iso"
    plotter_head.show(screenshot=str(head_only_png))
    print(f"Saved PyVista head-only plot: {head_only_png}")

    if SHOW_PYVISTA_HEAD_ONLY_INTERACTIVE:
        plotter_head2 = pv.Plotter()
        plotter_head2.add_mesh(grid, scalars=scalar_name, cmap=cmap,
                               show_edges=True, opacity=1.0)
        plotter_head2.add_scalar_bar(title=scalar_name)
        plotter_head2.add_axes()
        plotter_head2.show_grid()
        plotter_head2.camera_position = "iso"
        plotter_head2.show(title="Continuum model (Head only)")


def main() -> None:
    df = read_tensor_csv(CSV_PATH)
    k_perm, used_angle = select_permeability_tensor(df)
    k_hyd = permeability_to_hydraulic_conductivity(k_perm)

    k11, k22, k33, angle1 = principal_k_and_angle(k_hyd)

    print("Tensor selection:", TENSOR_SELECTION)
    if not np.isnan(used_angle):
        print(f"Closest angle from CSV: {used_angle:.1f} deg")
    print("Hydraulic conductivity tensor [m/s]:")
    print(k_hyd)
    print(f"Principal K [m/s]: k11={k11:.3e}, k22={k22:.3e}, k33={k33:.3e}")
    print(f"Horizontal major-axis angle1 [deg]: {angle1:.2f}")

    sim = build_continuum_model(k11, k22, k33, angle1)
    sim.write_simulation()
    print(f"Wrote MODFLOW 6 input to: {SIM_WS}")

    if RUN_MODEL:
        success, buff = sim.run_simulation(silent=True, report=True)
        if not success:
            raise RuntimeError("MODFLOW 6 failed:\n" + "\n".join(buff))
        print("Simulation completed.")
        if PLOT_2D_LAYER_MAP:
            quick_plot_last_layer_head(sim)
    else:
        print("RUN_MODEL=False -> input files written only.")

    if PLOT_WITH_PYVISTA:
        plot_continuum_with_pyvista(sim, k_hyd, angle1)


if __name__ == "__main__":
    main()
