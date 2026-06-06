"""Resolved Momentum Control (Kajita, Kanehiro, Kaneko, Fujiwara, Harada,
Yokoi & Hirukawa, IROS 2003).

The humanoid thread so far is a 1-D point mass: :mod:`lipm_walk`,
:mod:`capture_point`, :mod:`push_recovery`, :mod:`capturability` all collapse the
robot to its Center of Mass. **Resolved Momentum Control (RMC)** is the first
*whole-body* method here — it plans the motion of an articulated, free-floating
multibody so that its **total linear and angular momentum** take commanded values.

The key object is the **centroidal momentum matrix** ``A(q)``: the total spatial
momentum of the whole robot is *linear* in the generalized velocity,

    h = [Pₓ; P_y; L] = A(q)·q̇ ,   q̇ = [base ẋ, base ẏ, base θ̇, joint rates…],

where ``P`` is linear momentum (``= m·ṙ_C``, total mass times CoM velocity) and
``L`` is the angular momentum **about the CoM** (the *centroidal* angular momentum —
exactly the quantity the hip/flywheel strategy of :mod:`push_recovery` and the
reaction-mass model of :mod:`capturability` manipulate). This module builds ``A(q)``
for a planar (sagittal) articulated robot from each link's CoM and angular Jacobian.

**Resolving the motion.** Given a momentum reference ``h^ref`` together with task
constraints — a support foot pinned to the ground, a swing foot tracking a path —
stack the momentum and constraint Jacobians into ``B`` and invert:

    B q̇ = b ,   B = [A; J_support; J_swing] ,   b = [h^ref; 0; v_swing] ,

and solve the (typically redundant) system with the **pseudo-inverse**
``q̇ = Bᵀ(BBᵀ)⁻¹ b`` — the minimum-norm joint motion realizing the command, exactly
as the paper does with the inertia-matrix pseudo-inverse. The null-space
``N = I − B⁺B`` carries any extra task (posture) without disturbing the momentum.

Two faithful demonstrations from the paper: regulating the angular momentum to
**zero** while the CoM moves makes the body *counter-rotate* internally (the
centroidal angular momentum is held at zero by cancellation between the limbs —
the whole-body root of the reaction-mass strategy), and a **kick** drives a swing
foot along a commanded arc while the support foot stays pinned and the momentum
stays at its reference throughout.

Everything is pure Python — a tiny self-contained planar rigid-body kinematics and
a small dense linear-algebra kernel (Gaussian elimination, right pseudo-inverse).
The momentum matrix is *certified* against a finite-difference of the momentum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# tiny dense linear algebra (no numpy)
# --------------------------------------------------------------------------- #
def matvec(A, v):
    return [sum(A[r][c] * v[c] for c in range(len(v))) for r in range(len(A))]


def matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for k in range(m):
            a = Ai[k]
            if a:
                Bk = B[k]
                Ci = C[i]
                for j in range(p):
                    Ci[j] += a * Bk[j]
    return C


def transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def solve(A, b):
    """Solve the square system ``A x = b`` by Gaussian elimination w/ pivoting."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-15:
            raise ValueError("singular matrix in solve()")
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for j in range(c, n + 1):
            M[c][j] /= pv
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                for j in range(c, n + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][n] for i in range(n)]


def inverse(A):
    n = len(A)
    cols = []
    for k in range(n):
        e = [1.0 if i == k else 0.0 for i in range(n)]
        cols.append(solve(A, e))
    return transpose(cols)


def pinv_right(B):
    """Right pseudo-inverse ``B⁺ = Bᵀ(BBᵀ)⁻¹`` for a full-row-rank wide ``B``.

    Gives the minimum-norm solution of the underdetermined ``B q̇ = b``.
    """
    Bt = transpose(B)
    return matmul(Bt, inverse(matmul(B, Bt)))


def nullspace_projector(B):
    """``N = I − B⁺B``: projects onto motions that leave ``B q̇`` unchanged."""
    Bp = pinv_right(B)
    BpB = matmul(Bp, B)
    n = len(BpB)
    return [[(1.0 if i == j else 0.0) - BpB[i][j] for j in range(n)]
            for i in range(n)]


# --------------------------------------------------------------------------- #
# planar rigid-body model
# --------------------------------------------------------------------------- #
def _rot(a, v):
    c, s = math.cos(a), math.sin(a)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])


def _perp(v):
    """``ω ẑ × v`` for unit ``ω``: ``(−v_y, v_x)`` (planar revolute contribution)."""
    return (-v[1], v[0])


def _cross2(a, b):
    """Planar scalar cross product ``a × b = aₓb_y − a_ybₓ``."""
    return a[0] * b[1] - a[1] * b[0]


@dataclass
class Link:
    """A planar rigid link in the kinematic tree.

    ``parent`` is the index of the parent link (-1 for the free-floating base).
    ``anchor`` is the revolute joint position in the *parent's* frame; ``com`` is
    the link CoM in the *link's own* frame; ``mass`` and ``inertia`` are scalar.
    """

    parent: int
    anchor: tuple
    com: tuple
    mass: float
    inertia: float


@dataclass
class PlanarRobot:
    """A planar, free-floating articulated robot (sagittal-plane humanoid)."""

    links: list = field(default_factory=list)

    # ----- structure ------------------------------------------------------- #
    @property
    def n_joints(self):
        return len(self.links) - 1     # every non-base link adds one revolute DOF

    @property
    def ndof(self):
        return 3 + self.n_joints       # base (x, y, θ) + joints

    def joint_col(self, i):
        """Column of link ``i``'s joint rate in ``q`` (``i >= 1``)."""
        return 3 + (i - 1)

    def path_joints(self, i):
        """Joint-bearing links from the base up to link ``i`` (inclusive)."""
        out = []
        while i >= 1:
            out.append(i)
            i = self.links[i].parent
        return out

    def total_mass(self):
        return sum(L.mass for L in self.links)

    # ----- kinematics ------------------------------------------------------ #
    def forward_kinematics(self, q):
        """Return ``(phi, origin, com)`` world orientation / origin / CoM per link.

        Assumes parents have lower indices than children (true for the factories).
        """
        links = self.links
        phi = [0.0] * len(links)
        origin = [(0.0, 0.0)] * len(links)
        com = [(0.0, 0.0)] * len(links)
        phi[0] = q[2]
        origin[0] = (q[0], q[1])
        c0 = _rot(phi[0], links[0].com)
        com[0] = (q[0] + c0[0], q[1] + c0[1])
        for i in range(1, len(links)):
            L = links[i]
            p = L.parent
            phi[i] = phi[p] + q[self.joint_col(i)]
            aw = _rot(phi[p], L.anchor)
            ox, oy = origin[p][0] + aw[0], origin[p][1] + aw[1]
            origin[i] = (ox, oy)
            cw = _rot(phi[i], L.com)
            com[i] = (ox + cw[0], oy + cw[1])
        return phi, origin, com

    def com(self, q):
        """Whole-body Center of Mass position."""
        _, _, com = self.forward_kinematics(q)
        M = self.total_mass()
        cx = sum(self.links[i].mass * com[i][0] for i in range(len(self.links))) / M
        cy = sum(self.links[i].mass * com[i][1] for i in range(len(self.links))) / M
        return (cx, cy)

    def point_jacobian(self, q, link, local=(0.0, 0.0)):
        """``2×ndof`` Jacobian of a point fixed on ``link`` at offset ``local``.

        Also returns the point's world position.
        """
        phi, origin, _ = self.forward_kinematics(q)
        pw = (origin[link][0] + _rot(phi[link], local)[0],
              origin[link][1] + _rot(phi[link], local)[1])
        J = [[0.0] * self.ndof for _ in range(2)]
        J[0][0] = 1.0
        J[1][1] = 1.0
        d = (pw[0] - origin[0][0], pw[1] - origin[0][1])
        pb = _perp(d)
        J[0][2], J[1][2] = pb[0], pb[1]
        for j in self.path_joints(link):
            col = self.joint_col(j)
            dj = (pw[0] - origin[j][0], pw[1] - origin[j][1])
            pj = _perp(dj)
            J[0][col] += pj[0]
            J[1][col] += pj[1]
        return J, pw

    def com_jacobian(self, q, i):
        """``2×ndof`` Jacobian of link ``i``'s CoM."""
        J, _ = self.point_jacobian(q, i, self.links[i].com)
        return J

    def angular_jacobian(self, i):
        """``1×ndof`` Jacobian of link ``i``'s orientation rate ``φ̇_i``."""
        J = [0.0] * self.ndof
        J[2] = 1.0
        for j in self.path_joints(i):
            J[self.joint_col(j)] = 1.0
        return J

    # ----- momentum -------------------------------------------------------- #
    def centroidal_momentum_matrix(self, q):
        """The ``3×ndof`` centroidal momentum matrix ``A(q)``.

        Rows are ``[Pₓ; P_y; L_about_CoM]`` so that ``A(q)·q̇`` is the robot's
        total linear and (centroidal) angular momentum.
        """
        rc = self.com(q)
        _, _, com = self.forward_kinematics(q)
        nd = self.ndof
        A = [[0.0] * nd for _ in range(3)]
        for i in range(len(self.links)):
            m, I = self.links[i].mass, self.links[i].inertia
            Jc = self.com_jacobian(q, i)
            Jw = self.angular_jacobian(i)
            d = (com[i][0] - rc[0], com[i][1] - rc[1])
            for col in range(nd):
                A[0][col] += m * Jc[0][col]
                A[1][col] += m * Jc[1][col]
                A[2][col] += I * Jw[col] + m * _cross2(d, (Jc[0][col], Jc[1][col]))
        return A

    def momentum(self, q, qd):
        """Total momentum ``[Pₓ, P_y, L]`` for generalized velocity ``qd``."""
        return matvec(self.centroidal_momentum_matrix(q), qd)


# --------------------------------------------------------------------------- #
# model factory
# --------------------------------------------------------------------------- #
def make_humanoid():
    """A small planar humanoid: torso (base) + two 2-link legs + one arm."""
    return PlanarRobot(links=[
        Link(-1, (0.0, 0.0), (0.0, 0.10), 10.0, 0.30),   # 0 torso (base)
        Link(0, (0.0, -0.20), (0.0, -0.18), 3.0, 0.05),  # 1 right thigh
        Link(1, (0.0, -0.36), (0.0, -0.18), 2.0, 0.04),  # 2 right shank
        Link(0, (0.0, -0.20), (0.0, -0.18), 3.0, 0.05),  # 3 left thigh
        Link(3, (0.0, -0.36), (0.0, -0.18), 2.0, 0.04),  # 4 left shank
        Link(0, (0.10, 0.20), (0.0, -0.15), 1.5, 0.02),  # 5 arm
    ])


#: distal-tip offset of a shank link (foot point), in the shank's own frame.
SHANK_TIP = (0.0, -0.36)


@dataclass
class MomentumTask:
    """A resolved-momentum query: a momentum reference plus point-velocity tasks.

    ``constraints`` is a list of ``(link, local_offset, target_velocity)`` tuples,
    e.g. a pinned support foot ``(2, SHANK_TIP, (0.0, 0.0))`` or a swing foot
    tracking ``(4, SHANK_TIP, (vx, vy))``.
    """

    h_ref: tuple                       # (Px, Py, L)
    constraints: list = field(default_factory=list)


def _stacked(robot, q, task):
    A = robot.centroidal_momentum_matrix(q)
    B = [row[:] for row in A]
    b = list(task.h_ref)
    for link, local, vel in task.constraints:
        J, _ = robot.point_jacobian(q, link, local)
        B.extend(J)
        b.extend(vel)
    return B, b


def resolve_momentum(robot, q, task):
    """Minimum-norm generalized velocity realizing ``task`` at configuration ``q``.

    Stacks the centroidal momentum matrix with the constraint point-Jacobians and
    applies the right pseudo-inverse: ``q̇ = B⁺ b``.
    """
    B, b = _stacked(robot, q, task)
    return matvec(pinv_right(B), b)


def task_nullspace(robot, q, task):
    """Null-space projector of a momentum task (motions leaving ``B q̇`` fixed)."""
    B, _ = _stacked(robot, q, task)
    return nullspace_projector(B)


@dataclass
class MomentumTrajectory:
    """A rollout of resolved-momentum control over time."""

    q: list                            # configuration per step
    qd: list                           # generalized velocity per step
    momentum: list                     # [Px, Py, L] realized per step
    points: dict                       # {label: [world positions per step]}

    def max_abs_angular_momentum(self):
        return max(abs(h[2]) for h in self.momentum)

    def point_path(self, label):
        return self.points[label]


def simulate(robot, q0, task_fn, *, dt=0.01, steps=60, track=None):
    """Integrate resolved-momentum control: ``q̇ = resolve(...)``, Euler-step ``q``.

    ``task_fn(t, q) -> MomentumTask`` produces the command each step; ``track`` is
    an optional ``{label: (link, local)}`` of points to log along the motion.
    """
    track = track or {}
    q = list(q0)
    qs, qds, mom = [], [], []
    pts = {label: [] for label in track}
    for k in range(steps):
        t = k * dt
        task = task_fn(t, q)
        qd = resolve_momentum(robot, q, task)
        mom.append(robot.momentum(q, qd))
        qs.append(list(q))
        qds.append(list(qd))
        for label, (link, local) in track.items():
            _, pw = robot.point_jacobian(q, link, local)
            pts[label].append(pw)
        q = [q[i] + dt * qd[i] for i in range(robot.ndof)]
    for label, (link, local) in track.items():
        _, pw = robot.point_jacobian(q, link, local)
        pts[label].append(pw)
    return MomentumTrajectory(qs, qds, mom, pts)
