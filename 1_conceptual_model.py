"""Conceptual MODFLOW 6 model (100 x 100 x 100 m) with PyVista visualization.

This script is intentionally easy to edit:
- geometry and grid settings are in ConceptualModelConfig
- initial conditions are controlled in `initial_condition_mode`
- model can be built, written, and run with FloPy/MODFLOW 6

Note:
`cells_per_meter=10` means 1000 x 1000 x 1000 cells (= 1e9 cells), which is
far beyond practical memory limits for normal machines. A safeguard is included.
"""

=======
from __future__ import annotations
>>>>>>> cd9c77c66deacae68a3002a73a6033c9a9dcc31a

from dataclasses import dataclass
from pathlib import Path

import flopy
import numpy as np
import pyvista as pv


@dataclass
class ConceptualModelConfig:
    # Domain size [m]
    length_x: float = 100.0
    length_y: float = 100.0
    length_z: float = 100.0

    # Grid resolution: number of cells per meter
    # 1 cell/m gives a 100 x 100 x 100 grid for a 100 m cube.
    cells_per_meter: int = 1

    # Model top elevation [m]
    top: float = 0.0

    # Model origin in plan view [m]
    x_origin: float = 0.0
    y_origin: float = 0.0

    # Initial condition controls
    # "uniform" -> use initial_head_uniform
    # "linear_x" -> linear gradient from initial_head_left to initial_head_right
    initial_condition_mode: str = "uniform"
    initial_head_uniform: float = 80.0
    initial_head_left: float = 100.0
    initial_head_right: float = 50.0

    # Hydraulic properties
    hk: float = 10.0
    vk: float = 10.0

    # Boundary controls
    use_chd_left: bool = True
    use_chd_right: bool = True

    # Runtime controls
    simulation_name: str = "conceptual"
    mf6_exe_name: str = "mf6"
    run_simulation: bool = True
    max_cells_for_run: int = 2_000_000

    @property
    def ncol(self) -> int:
        return int(self.length_x * self.cells_per_meter)

    @property
    def nrow(self) -> int:
        return int(self.length_y * self.cells_per_meter)

    @property
    def nlay(self) -> int:
        return int(self.length_z * self.cells_per_meter)

    @property
    def delr(self) -> float:
        return 1.0 / self.cells_per_meter

    @property
    def delc(self) -> float:
        return 1.0 / self.cells_per_meter

    @property
    def delv(self) -> float:
        return 1.0 / self.cells_per_meter

    @property
    def total_cells(self) -> int:
        return self.nlay * self.nrow * self.ncol

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            self.x_origin + 0.5 * self.length_x,
            self.y_origin + 0.5 * self.length_y,
            self.top - 0.5 * self.length_z,
        )


def build_initial_heads(cfg: ConceptualModelConfig, x_centers: np.ndarray) -> np.ndarray:
    """Create starting heads as (nlay, nrow, ncol)."""
    if cfg.initial_condition_mode == "uniform":
        return np.full((cfg.nlay, cfg.nrow, cfg.ncol), cfg.initial_head_uniform, dtype=float)

    if cfg.initial_condition_mode == "linear_x":
        x_norm = np.clip((x_centers - cfg.x_origin) / cfg.length_x, 0.0, 1.0)
        head_line = cfg.initial_head_left + (cfg.initial_head_right - cfg.initial_head_left) * x_norm
        head_2d = np.tile(head_line, (cfg.nrow, 1))
        return np.tile(head_2d[np.newaxis, :, :], (cfg.nlay, 1, 1))

    raise ValueError(
        "initial_condition_mode must be 'uniform' or 'linear_x'"
    )


def build_flopy_model(cfg: ConceptualModelConfig, workspace: Path) -> tuple[flopy.mf6.MFSimulation, flopy.mf6.ModflowGwf]:
    """Build a structured MODFLOW 6 model in FloPy."""
    sim = flopy.mf6.MFSimulation(
        sim_name=cfg.simulation_name,
        sim_ws=str(workspace),
        exe_name=cfg.mf6_exe_name,
    )

    flopy.mf6.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=1,
        perioddata=[(1.0, 1, 1.0)],
    )
    flopy.mf6.ModflowIms(sim, pname="ims", complexity="SIMPLE")

    gwf = flopy.mf6.ModflowGwf(sim, modelname=cfg.simulation_name, save_flows=True)

    botm = cfg.top - np.arange(1, cfg.nlay + 1, dtype=float) * cfg.delv
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=cfg.nlay,
        nrow=cfg.nrow,
        ncol=cfg.ncol,
        delr=cfg.delr,
        delc=cfg.delc,
        top=cfg.top,
        botm=botm,
        xorigin=cfg.x_origin,
        yorigin=cfg.y_origin,
    )

    x_centers = cfg.x_origin + (np.arange(cfg.ncol, dtype=float) + 0.5) * cfg.delr
    strt = build_initial_heads(cfg, x_centers)
    flopy.mf6.ModflowGwfic(gwf, strt=strt)

    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=1,
        k=cfg.hk,
        k33=cfg.vk,
        save_flows=True,
    )

    chd_spd = []
    if cfg.use_chd_left:
        for k in range(cfg.nlay):
            for i in range(cfg.nrow):
                chd_spd.append(((k, i, 0), cfg.initial_head_left))

    if cfg.use_chd_right:
        for k in range(cfg.nlay):
            for i in range(cfg.nrow):
                chd_spd.append(((k, i, cfg.ncol - 1), cfg.initial_head_right))

    if chd_spd:
        flopy.mf6.ModflowGwfchd(
            gwf,
            stress_period_data=chd_spd,
            save_flows=True,
        )

    flopy.mf6.ModflowGwfoc(
        gwf,
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "LAST")],
        head_filerecord=[f"{cfg.simulation_name}.hds"],
        budget_filerecord=[f"{cfg.simulation_name}.cbb"],
    )

    return sim, gwf


def build_visual_cube(cfg: ConceptualModelConfig) -> pv.PolyData:
    """Transparent conceptual cube for quick visualization."""
    cx, cy, cz = cfg.center
    return pv.Cube(
        center=(cx, cy, cz),
        x_length=cfg.length_x,
        y_length=cfg.length_y,
        z_length=cfg.length_z,
    )


def build_surface_cell_lines(cfg: ConceptualModelConfig) -> pv.PolyData:
    """Create line segments for cell boundaries on all six cube faces."""
    x0 = cfg.x_origin
    x1 = cfg.x_origin + cfg.length_x
    y0 = cfg.y_origin
    y1 = cfg.y_origin + cfg.length_y
    z_top = cfg.top
    z_bot = cfg.top - cfg.length_z

    xs = np.linspace(x0, x1, cfg.ncol + 1)
    ys = np.linspace(y0, y1, cfg.nrow + 1)
    zs = np.linspace(z_bot, z_top, cfg.nlay + 1)

    points: list[tuple[float, float, float]] = []
    lines: list[int] = []

    def add_segment(p0: tuple[float, float, float], p1: tuple[float, float, float]) -> None:
        i0 = len(points)
        points.append(p0)
        i1 = len(points)
        points.append(p1)
        lines.extend([2, i0, i1])

    # Top and bottom faces (xy).
    for z in (z_top, z_bot):
        for y in ys:
            add_segment((x0, y, z), (x1, y, z))
        for x in xs:
            add_segment((x, y0, z), (x, y1, z))

    # Front and back faces (xz).
    for y in (y0, y1):
        for z in zs:
            add_segment((x0, y, z), (x1, y, z))
        for x in xs:
            add_segment((x, y, z_bot), (x, y, z_top))

    # Left and right faces (yz).
    for x in (x0, x1):
        for z in zs:
            add_segment((x, y0, z), (x, y1, z))
        for y in ys:
            add_segment((x, y, z_bot), (x, y, z_top))

    mesh = pv.PolyData()
    mesh.points = np.asarray(points, dtype=float)
    mesh.lines = np.asarray(lines, dtype=np.int64)
    return mesh


def apply_equal_axes(plotter: pv.Plotter, bounds: tuple[float, float, float, float, float, float]) -> None:
    """Force an equal-axis reference box in PyVista."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    zmid = 0.5 * (zmin + zmax)

    max_len = max(xmax - xmin, ymax - ymin, zmax - zmin)
    equal_box = pv.Cube(center=(xmid, ymid, zmid), x_length=max_len, y_length=max_len, z_length=max_len)
    plotter.add_mesh(equal_box.outline(), color="black", line_width=1.0)


def plot_conceptual_model(cfg: ConceptualModelConfig) -> None:
    """Plot transparent conceptual cube with center marker and coordinate axes."""
    cube = build_visual_cube(cfg)
    cell_lines = build_surface_cell_lines(cfg)
    center_marker = pv.Sphere(radius=1.0, center=cfg.center)

    p = pv.Plotter()
    p.add_mesh(cube, color="lightskyblue", opacity=0.25, show_edges=True)
    p.add_mesh(cell_lines, color="midnightblue", line_width=0.7, opacity=0.8)
    p.add_mesh(center_marker, color="crimson")
    p.add_axes()
    apply_equal_axes(p, cube.bounds)
    p.view_isometric()
    p.show(title="Conceptual Model (Transparent Cube)")


def main() -> None:
    cfg = ConceptualModelConfig()

    print("Conceptual model summary")
    print(f"Domain: {cfg.length_x} x {cfg.length_y} x {cfg.length_z} m")
    print(f"Resolution: {cfg.cells_per_meter} cells/m")
    print(f"Grid: nlay={cfg.nlay}, nrow={cfg.nrow}, ncol={cfg.ncol}")
    print(f"Total cells: {cfg.total_cells:,}")
    print(f"Center: {cfg.center}")

    workspace = Path.cwd() / f"{cfg.simulation_name}_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    sim, _gwf = build_flopy_model(cfg, workspace)
    sim.write_simulation()
    print(f"Wrote simulation files to: {workspace}")

    if cfg.run_simulation:
        if cfg.total_cells > cfg.max_cells_for_run:
            raise RuntimeError(
                "Configured grid is too large to run on normal hardware. "
                f"cells_per_meter={cfg.cells_per_meter} gives {cfg.total_cells:,} cells. "
                "Lower cells_per_meter (e.g. 1 or 2) or increase max_cells_for_run deliberately."
            )

        success, buff = sim.run_simulation(silent=False, report=True)
        if not success:
            raise RuntimeError("MODFLOW 6 run failed:\n" + "\n".join(str(x) for x in buff))
        print("Simulation completed successfully.")

    plot_conceptual_model(cfg)


if __name__ == "__main__":
    main()
