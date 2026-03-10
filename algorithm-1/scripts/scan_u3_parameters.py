"""Find the U3 gate furthest from Clifford and visualise it on the Bloch sphere.

U3(theta, phi, lam) covers all of SU(2), so this is equivalent to searching
over all single-qubit unitaries.  Uses differential evolution to minimise
p_acc, then maps the optimal gate to SO(3) via the adjoint representation.

The adjoint map R[i,j] = (1/2) Re Tr(sigma_i U sigma_j U†) is exactly how
U acts on Bloch-sphere coordinates (⟨X⟩, ⟨Y⟩, ⟨Z⟩), so drawing R on the
Bloch sphere is the correct and natural picture.  Global phase of U3 cancels
in U sigma U†, so R depends only on the SU(2) equivalence class of U.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from qiskit.circuit.library import U3Gate
from scipy.optimize import differential_evolution, minimize_scalar

from lib.expected_acceptance_probability import expected_acceptance_probability

N_QUBITS = 1

# Pauli matrices X, Y, Z (indices 0, 1, 2)
SIGMA = [
    np.array([[0, 1], [1, 0]], dtype=complex),    # X
    np.array([[0, -1j], [1j, 0]], dtype=complex),  # Y
    np.array([[1, 0], [0, -1]], dtype=complex),    # Z
]


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------

def u3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    return U3Gate(theta, phi, lam).to_matrix()


def p_acc_for(theta: float, phi: float, lam: float) -> float:
    return expected_acceptance_probability(u3_matrix(theta, phi, lam), N_QUBITS)


# ---------------------------------------------------------------------------
# Global optimisation
# ---------------------------------------------------------------------------

def find_global_minimum() -> tuple[float, float, float, float]:
    """Use differential evolution to find (theta, phi, lam) minimising p_acc.

    Returns (theta, phi, lam, p_acc_min).
    """
    bounds = [(0, np.pi), (0, 2 * np.pi), (0, 2 * np.pi)]

    result = differential_evolution(
        lambda p: p_acc_for(p[0], p[1], p[2]),
        bounds,
        seed=42,
        maxiter=1000,
        tol=1e-9,
        polish=True,
        workers=1,
    )

    theta, phi, lam = result.x
    return float(theta), float(phi), float(lam), float(result.fun)


# ---------------------------------------------------------------------------
# SU(2) -> SO(3) via adjoint representation
# ---------------------------------------------------------------------------

def su2_to_so3(U: np.ndarray) -> np.ndarray:
    """Map an SU(2) matrix to the corresponding SO(3) rotation.

    R[i, j] = (1/2) Re Tr(sigma_i  U  sigma_j  U†)
    where sigma_0=X, sigma_1=Y, sigma_2=Z.
    """
    U_dag = U.conj().T
    R = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            R[i, j] = 0.5 * np.trace(SIGMA[i] @ U @ SIGMA[j] @ U_dag).real
    return R


def so3_axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Extract rotation axis (unit vector) and angle from an SO(3) matrix."""
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))

    if abs(angle) < 1e-10:
        return np.array([0.0, 0.0, 1.0]), 0.0

    if abs(angle - np.pi) < 1e-6:
        sym = R + R.T
        col_norms = np.linalg.norm(sym, axis=0)
        axis = sym[:, np.argmax(col_norms)]
        return axis / np.linalg.norm(axis), angle

    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]])
    return axis / np.linalg.norm(axis), angle


# ---------------------------------------------------------------------------
# Single-qubit Clifford group
# ---------------------------------------------------------------------------

def clifford_images_of_zero() -> tuple[np.ndarray, list[str]]:
    """Return Bloch sphere images of |0⟩ under all 24 single-qubit Cliffords.

    The single-qubit Clifford group acts on the Bloch sphere as the rotation
    group of the octahedron.  The orbit of |0⟩ = +ẑ under these 24 rotations
    is exactly the 6 octahedron vertices {±x̂, ±ŷ, ±ẑ}, each reached by 4
    distinct Clifford gates.

    Uses BFS from identity with generators H and S to enumerate all 24
    elements (up to global phase), then maps each via su2_to_so3.

    Returns (points, labels) where points has shape (6, 3) — one row per
    unique Bloch sphere vertex — and labels are the standard state names.
    """
    sq2 = np.sqrt(2)
    H_mat = np.array([[1, 1], [1, -1]], dtype=complex) / sq2
    S_mat = np.array([[1, 0], [0, 1j]], dtype=complex)

    def same_up_to_phase(U: np.ndarray, V: np.ndarray) -> bool:
        M = U.conj().T @ V  # = e^{iφ} I if equal up to phase
        if abs(M[0, 1]) > 1e-8 or abs(M[1, 0]) > 1e-8:
            return False
        return abs(abs(M[0, 0]) - 1) < 1e-8 and abs(M[0, 0] - M[1, 1]) < 1e-8

    cliffords: list[np.ndarray] = [np.eye(2, dtype=complex)]
    queue: list[np.ndarray] = [np.eye(2, dtype=complex)]

    while queue and len(cliffords) < 24:
        U = queue.pop(0)
        for G in (H_mat, S_mat):
            V = G @ U
            if not any(same_up_to_phase(V, C) for C in cliffords):
                cliffords.append(V)
                queue.append(V)

    assert len(cliffords) == 24, f"Expected 24 Cliffords, got {len(cliffords)}"

    z_hat = np.array([0.0, 0.0, 1.0])
    raw_images = np.array([su2_to_so3(C) @ z_hat for C in cliffords])

    # Deduplicate: group the 24 images into 6 unique vertices
    vertex_labels = {
        ( 0,  0,  1): "$|0\\rangle$  (+z)",
        ( 0,  0, -1): "$|1\\rangle$  (-z)",
        ( 1,  0,  0): "$|{+}\\rangle$  (+x)",
        (-1,  0,  0): "$|{-}\\rangle$  (-x)",
        ( 0,  1,  0): "$|{+i}\\rangle$  (+y)",
        ( 0, -1,  0): "$|{-i}\\rangle$  (-y)",
    }

    unique: dict[tuple[int, int, int], int] = {}  # key -> count
    for pt in raw_images:
        key = tuple(round(v) for v in pt)
        unique[key] = unique.get(key, 0) + 1

    points = np.array([list(k) for k in unique], dtype=float)
    labels = [vertex_labels[k] for k in unique]
    return points, labels


# ---------------------------------------------------------------------------
# Sphere colouring
# ---------------------------------------------------------------------------

def compute_p_acc_sphere(steps: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each direction on the Bloch sphere, compute min_lam p_acc(theta, phi, lam).

    The colour at direction (theta, phi) is the lowest p_acc achievable by any
    gate that sends |0⟩ to that direction (i.e. optimised over lam).  This
    guarantees the destination of |0⟩ under the globally optimal gate sits at
    the darkest point on the sphere.

    Returns (X, Y, Z, P) arrays of shape (steps, 2*steps).
    """
    thetas = np.linspace(0, np.pi, steps)
    phis   = np.linspace(0, 2 * np.pi, 2 * steps)

    PHI, THETA = np.meshgrid(phis, thetas)
    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    P = np.zeros_like(THETA)
    for i in range(steps):
        for j in range(2 * steps):
            res = minimize_scalar(
                lambda lam, t=thetas[i], p=phis[j]: p_acc_for(t, p, lam),
                bounds=(0, 2 * np.pi),
                method="bounded",
            )
            P[i, j] = res.fun

    return X, Y, Z, P


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _sphere_wireframe(ax: plt.Axes, alpha: float = 0.12) -> None:
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="lightblue", alpha=alpha, linewidth=0)
    for lat in np.linspace(-np.pi / 2, np.pi / 2, 7):
        ax.plot(np.cos(lat) * np.cos(u), np.cos(lat) * np.sin(u),
                np.sin(lat) * np.ones_like(u), "gray", lw=0.4, alpha=0.4)
    for lon in np.linspace(0, np.pi, 7):
        ax.plot(np.cos(lon) * np.sin(v), np.sin(lon) * np.sin(v),
                np.cos(v), "gray", lw=0.4, alpha=0.4)


def _arrow(ax: plt.Axes, vec: np.ndarray, color: str, label: str,
           scale: float = 1.0) -> None:
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2],
              length=scale, color=color, arrow_length_ratio=0.12, linewidth=2)
    tip = vec * scale
    ax.text(tip[0] * 1.15, tip[1] * 1.15, tip[2] * 1.15, label,
            color=color, fontsize=10, fontweight="bold")


def visualize_rotation(
    axis: np.ndarray,
    angle: float,
    R: np.ndarray,
    best_theta: float,
    best_phi: float,
    best_lam: float,
    best_p_acc: float,
    sphere_steps: int,
    out_path: Path,
) -> None:
    """Draw the SO(3) rotation on a p_acc-coloured Bloch sphere."""
    fig = plt.figure(figsize=(9, 7))
    ax: plt.Axes = fig.add_subplot(111, projection="3d")

    # Coloured sphere: min_lam p_acc at each direction
    print("  Computing p_acc sphere (min over lam)...", flush=True)
    X, Y, Z, P = compute_p_acc_sphere(sphere_steps)
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=P.min(), vmax=1.0)
    ax.plot_surface(X, Y, Z, facecolors=cmap(norm(P)),
                    alpha=0.75, linewidth=0, antialiased=True)

    # Lat/lon grid lines on the coloured surface
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    for lat in np.linspace(-np.pi / 2, np.pi / 2, 7):
        ax.plot(np.cos(lat) * np.cos(u), np.cos(lat) * np.sin(u),
                np.sin(lat) * np.ones_like(u), "white", lw=0.4, alpha=0.3)
    for lon in np.linspace(0, np.pi, 7):
        ax.plot(np.cos(lon) * np.sin(v), np.sin(lon) * np.sin(v),
                np.cos(v), "white", lw=0.4, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.05)
    cbar.set_label("min$_\\lambda\\,p_{\\rm acc}$  (1.0 = Clifford)", fontsize=9)

    # Clifford images of |0⟩ — the 6 octahedron vertices
    # Lifted to r=1.03 so they sit in front of the sphere surface
    clifford_pts, clifford_labels = clifford_images_of_zero()
    cp = clifford_pts * 1.03
    ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2],
               color="gold", s=80, depthshade=False,
               label="Clifford images of $|0\\rangle$ (6 vertices, 4 gates each)")
    for pt, lbl in zip(clifford_pts, clifford_labels, strict=True):
        offset = pt * 1.22
        ax.text(offset[0], offset[1], offset[2], lbl,
                fontsize=7, color="goldenrod", ha="center", va="center")

    # Rotation axis
    _arrow(ax, axis, "crimson", "axis $\\hat{n}$", scale=1.25)
    ax.plot([-axis[0] * 1.3, axis[0] * 1.3],
            [-axis[1] * 1.3, axis[1] * 1.3],
            [-axis[2] * 1.3, axis[2] * 1.3],
            color="crimson", lw=1, ls="--", alpha=0.5)

    # |0⟩ state = north pole = z-axis; fallback to x if axis is parallel to z
    z_hat = np.array([0.0, 0.0, 1.0])
    v0 = z_hat if abs(np.dot(z_hat, axis)) < 0.99 else np.array([1.0, 0.0, 0.0])
    v0 = v0 - np.dot(v0, axis) * axis
    v0 /= np.linalg.norm(v0)

    _arrow(ax, v0, "royalblue", "$|0\\rangle$")

    v1 = R @ v0
    _arrow(ax, v1, "forestgreen", "$U|0\\rangle$")

    # Mark the landing point lifted to r=1.03 to sit above the surface
    v1s = v1 * 1.03
    ax.scatter([v1s[0]], [v1s[1]], [v1s[2]], color="lime", s=120,
               depthshade=False, label=f"$U|0\\rangle$ landing point  ($p_{{\\rm acc}}$={best_p_acc:.4f})")

    # Arc tracing the rotation, lifted to r=1.02
    perp = np.cross(axis, v0)
    perp /= np.linalg.norm(perp)
    t_vals = np.linspace(0, angle, 120)
    arc = 1.02 * (np.outer(np.cos(t_vals), v0) + np.outer(np.sin(t_vals), perp))
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="darkorange", lw=2.5,
            label=f"rotation arc ({np.degrees(angle):.1f}°)")
    mid = len(t_vals) // 2
    mp = arc[mid] * 1.18
    ax.text(mp[0], mp[1], mp[2], f"  {np.degrees(angle):.1f}°",
            color="darkorange", fontsize=9)

    # Coordinate axes
    for vec, lbl in [(np.array([1, 0, 0]), "x"), (np.array([0, 1, 0]), "y"),
                     (np.array([0, 0, 1]), "z / $|0\\rangle$")]:
        ax.plot([0, 1.5*vec[0]], [0, 1.5*vec[1]], [0, 1.5*vec[2]],
                "k--", lw=0.7, alpha=0.3)
        ax.text(1.6*vec[0], 1.6*vec[1], 1.6*vec[2], lbl,
                ha="center", va="center", fontsize=9, color="gray")

    axis_str = f"({axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f})"
    title = (
        f"Most non-Clifford U3: θ={best_theta/np.pi:.4f}π, "
        f"φ={best_phi/np.pi:.4f}π, λ={best_lam/np.pi:.4f}π\n"
        f"Rotation axis $\\hat{{n}}$ = {axis_str},  "
        f"angle = {np.degrees(angle):.2f}°\n"
        f"$p_{{\\rm acc}}$ = {best_p_acc:.6f}  (Clifford → 1.0)"
    )
    ax.set_title(title, fontsize=10, pad=12)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_zlim(-1.4, 1.4)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("algorithm-1/results/u3_rotation_bloch.png"),
        help="Output path for the figure",
    )
    parser.add_argument(
        "--sphere-steps",
        type=int,
        default=50,
        help="Grid resolution for the p_acc sphere colouring (default: 50)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Global optimisation
    # ------------------------------------------------------------------
    print("[1/2] Running global optimisation (differential evolution)...")
    best_theta, best_phi, best_lam, best_p_acc = find_global_minimum()

    print(f"\n  theta = {best_theta:.6f} rad  ({best_theta/np.pi:.6f} * pi)")
    print(f"  phi   = {best_phi:.6f} rad  ({best_phi/np.pi:.6f} * pi)")
    print(f"  lam   = {best_lam:.6f} rad  ({best_lam/np.pi:.6f} * pi)")
    print(f"  p_acc = {best_p_acc:.8f}")

    # ------------------------------------------------------------------
    # 2. SO(3) analysis and visualisation
    # ------------------------------------------------------------------
    print("\n[2/2] Computing SO(3) rotation and drawing Bloch sphere...")

    U_best = u3_matrix(best_theta, best_phi, best_lam)
    R = su2_to_so3(U_best)
    axis, angle = so3_axis_angle(R)

    print("\n  SO(3) matrix R =")
    for row in R:
        print("    " + "  ".join(f"{v:+.6f}" for v in row))

    print(f"\n  Rotation axis  : ({axis[0]:+.6f}, {axis[1]:+.6f}, {axis[2]:+.6f})")
    print(f"  Rotation angle : {angle:.6f} rad  ({np.degrees(angle):.4f}°)"
          f"  ({angle/np.pi:.6f} * pi)")

    det = np.linalg.det(R)
    orth_err = np.max(np.abs(R @ R.T - np.eye(3)))
    print(f"\n  det(R) = {det:.6f}  (should be +1)")
    print(f"  ||R Rᵀ - I||∞ = {orth_err:.2e}  (should be ~0)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    visualize_rotation(axis, angle, R,
                       best_theta, best_phi, best_lam, best_p_acc,
                       args.sphere_steps, args.out)


if __name__ == "__main__":
    main()
