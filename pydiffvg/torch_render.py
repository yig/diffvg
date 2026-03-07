"""Pure PyTorch implementation of the diffvg differentiable vector graphics renderer.

Replaces the C++/CUDA backend, enabling GPU acceleration on any PyTorch-supported
device (CUDA, MPS/Metal, ROCm, etc.).
"""

import torch
import math
from typing import List, Optional, Tuple
from enum import IntEnum


# ============================================================================
# Enums (matching C++ enums)
# ============================================================================

class ShapeType(IntEnum):
    circle = 0
    ellipse = 1
    path = 2
    rect = 3

class ColorType(IntEnum):
    constant = 0
    linear_gradient = 1
    radial_gradient = 2

class FilterType(IntEnum):
    box = 0
    tent = 1
    parabolic = 2 # radial parabolic
    hann = 3 # Hann


# ============================================================================
# Polynomial Solvers (vectorized)
# ============================================================================

def solve_quadratic(a, b, c):
    """Solve ax^2 + bx + c = 0 for batched inputs.

    Args:
        a, b, c: tensors of shape [...] (any batch dims)

    Returns:
        roots: tensor of shape [..., 2]
        valid: boolean tensor of shape [..., 2]
    """
    discrim = b * b - 4 * a * c
    has_roots = discrim >= 0
    safe_discrim = torch.where(has_roots, discrim, torch.zeros_like(discrim))
    root_discrim = torch.sqrt(safe_discrim)

    q = torch.where(b < 0,
                    -0.5 * (b - root_discrim),
                    -0.5 * (b + root_discrim))
    safe_a = torch.where(a.abs() < 1e-10, torch.ones_like(a), a)
    safe_q = torch.where(q.abs() < 1e-10, torch.ones_like(q), q)

    t0 = q / safe_a
    t1 = c / safe_q

    # Sort
    t0_out = torch.minimum(t0, t1)
    t1_out = torch.maximum(t0, t1)

    roots = torch.stack([t0_out, t1_out], dim=-1)
    valid = has_roots.unsqueeze(-1).expand_as(roots)
    # Invalidate if a is near-zero (degenerate)
    linear = a.abs() < 1e-10
    if linear.any():
        safe_b = torch.where(b.abs() < 1e-10, torch.ones_like(b), b)
        linear_root = -c / safe_b
        linear_valid = (b.abs() >= 1e-10) & linear
        roots = torch.where(linear.unsqueeze(-1),
                           torch.stack([linear_root, linear_root], dim=-1),
                           roots)
        valid = torch.where(linear.unsqueeze(-1),
                           torch.stack([linear_valid, torch.zeros_like(linear_valid)], dim=-1),
                           valid)
    return roots, valid


def solve_cubic(a, b, c, d):
    """Solve ax^3 + bx^2 + cx + d = 0 for batched inputs using Cardano's formula.

    Args:
        a, b, c, d: tensors of shape [...]

    Returns:
        roots: tensor of shape [..., 3]
        num_roots: integer tensor of shape [...]
    """
    device = a.device
    # Use double precision for numerical stability. Unfortunately, MPS doesn't support it.
    dtype = torch.float32 if device.type == 'mps' else torch.float64
    a = a.to(dtype)
    b = b.to(dtype)
    c = c.to(dtype)
    d = d.to(dtype)

    shape = a.shape
    roots = torch.zeros(*shape, 3, device=device, dtype=dtype)
    num_roots = torch.zeros(shape, device=device, dtype=torch.int32)

    # Handle degenerate case: a near zero -> solve quadratic
    degen = a.abs() < 1e-6
    if degen.any():
        qr, qv = solve_quadratic(b[degen], c[degen], d[degen])
        qr = qr.to(dtype)
        roots[degen, :2] = qr
        num_roots[degen] = qv.sum(dim=-1).to(torch.int32)

    # Non-degenerate cubic
    nd = ~degen
    if nd.any():
        a_nd = a[nd]
        bn = b[nd] / a_nd
        cn = c[nd] / a_nd
        dn = d[nd] / a_nd

        Q = (bn * bn - 3 * cn) / 9.0
        R = (2 * bn * bn * bn - 9 * bn * cn + 27 * dn) / 54.0

        three_real = (R * R) < (Q * Q * Q)

        # Case 1: three real roots
        tr = three_real
        if tr.any():
            Q_tr = Q[tr]
            R_tr = R[tr]
            bn_tr = bn[tr]
            theta = torch.acos(torch.clamp(R_tr / torch.sqrt(Q_tr * Q_tr * Q_tr),
                                          -1.0, 1.0))
            sq_Q = torch.sqrt(Q_tr)
            r0 = -2.0 * sq_Q * torch.cos(theta / 3.0) - bn_tr / 3.0
            r1 = -2.0 * sq_Q * torch.cos((theta + 2 * math.pi) / 3.0) - bn_tr / 3.0
            r2 = -2.0 * sq_Q * torch.cos((theta - 2 * math.pi) / 3.0) - bn_tr / 3.0

            # Build index into the nd positions for three_real
            nd_indices = torch.where(nd)[0]
            tr_indices = nd_indices[tr]
            roots[tr_indices, 0] = r0
            roots[tr_indices, 1] = r1
            roots[tr_indices, 2] = r2
            num_roots[tr_indices] = 3

        # Case 2: one real root
        one = ~three_real
        if one.any():
            Q_one = Q[one]
            R_one = R[one]
            bn_one = bn[one]
            R2_Q3 = R_one * R_one - Q_one * Q_one * Q_one
            safe_R2_Q3 = torch.clamp(R2_Q3, min=0.0)

            pos_R = R_one > 0
            A = torch.where(
                pos_R,
                -(R_one + torch.sqrt(safe_R2_Q3)).sign() *
                    (R_one + torch.sqrt(safe_R2_Q3)).abs().pow(1.0/3.0),
                (-R_one + torch.sqrt(safe_R2_Q3)).sign() *
                    (-R_one + torch.sqrt(safe_R2_Q3)).abs().pow(1.0/3.0)
            )
            safe_A = torch.where(A.abs() > 1e-6, A, torch.ones_like(A))
            B = torch.where(A.abs() > 1e-6, Q_one / safe_A, torch.zeros_like(A))
            r0 = (A + B) - bn_one / 3.0

            nd_indices = torch.where(nd)[0]
            one_indices = nd_indices[one]
            roots[one_indices, 0] = r0
            num_roots[one_indices] = 1

    return roots.to(a.dtype if a.dtype != dtype else torch.float32), num_roots


# ============================================================================
# Path Segment Extraction
# ============================================================================

def extract_path_segments(points, num_control_points, is_closed):
    """Extract segments from a path and return grouped by type.

    Returns:
        lines: list of (i0, i1) index pairs into points
        quadratics: list of (i0, i1, i2) index triples
        cubics: list of (i0, i1, i2, i3) index quads
        All indices refer to rows of `points` tensor [N, 2].
    """
    n_pts = points.shape[0]
    n_segs = num_control_points.shape[0]
    lines = []
    quadratics = []
    cubics = []

    point_id = 0
    for seg_i in range(n_segs):
        ncp = num_control_points[seg_i].item()
        if ncp == 0:
            i0 = point_id
            i1 = (point_id + 1) % n_pts
            lines.append((i0, i1))
        elif ncp == 1:
            i0 = point_id
            i1 = point_id + 1
            i2 = (point_id + 2) % n_pts
            quadratics.append((i0, i1, i2))
        elif ncp == 2:
            i0 = point_id
            i1 = point_id + 1
            i2 = point_id + 2
            i3 = (point_id + 3) % n_pts
            cubics.append((i0, i1, i2, i3))
        point_id += ncp + 1
    return lines, quadratics, cubics


# ============================================================================
# Winding Number Computation (vectorized across query points)
# ============================================================================

def _winding_line(p0, p1, pts):
    """Winding number contribution from a line segment for all query points.

    Args:
        p0, p1: [2] tensors (segment endpoints)
        pts: [N, 2] query points

    Returns: [N] integer winding contributions
    """
    # Solve: pt.y = p0.y + t*(p1.y - p0.y) for t
    dy = p1[1] - p0[1]
    winding = torch.zeros(pts.shape[0], device=pts.device, dtype=torch.int32)
    # Skip horizontal lines
    if dy.abs() < 1e-10:
        return winding
    t = (pts[:, 1] - p0[1]) / dy
    valid = (t >= 0) & (t <= 1)
    # t' = p0.x - pt.x + t * (p1.x - p0.x)
    tp = p0[0] - pts[:, 0] + t * (p1[0] - p0[0])
    hit = valid & (tp >= 0)
    sign = torch.where(dy > 0,
                       torch.ones_like(winding),
                       -torch.ones_like(winding))
    winding = torch.where(hit, sign, winding)
    return winding


def _winding_quadratic(p0, p1, p2, pts):
    """Winding number contribution from a quadratic Bezier for all query points.

    The curve is (1-t)^2*p0 + 2*(1-t)*t*p1 + t^2*p2
    = (p0-2p1+p2)t^2 + (-2p0+2p1)t + p0
    """
    a_coeff = p0[1] - 2*p1[1] + p2[1]
    b_coeff = -2*p0[1] + 2*p1[1]
    c_coeff = p0[1]  # will subtract pts[:, 1]

    winding = torch.zeros(pts.shape[0], device=pts.device, dtype=torch.int32)

    # For each query point, solve quadratic
    c_val = c_coeff - pts[:, 1]
    roots, valid = solve_quadratic(
        a_coeff.expand(pts.shape[0]),
        b_coeff.expand(pts.shape[0]),
        c_val
    )

    ax = p0[0] - 2*p1[0] + p2[0]
    bx = -2*p0[0] + 2*p1[0]

    for j in range(2):
        t = roots[:, j]
        v = valid[:, j] & (t >= 0) & (t <= 1)
        tp = ax * t * t + bx * t + p0[0] - pts[:, 0]
        hit = v & (tp >= 0)
        # Derivative at t: 2*a_coeff*t + b_coeff
        deriv = 2 * a_coeff * t + b_coeff
        sign = torch.where(deriv > 0,
                          torch.ones(pts.shape[0], device=pts.device, dtype=torch.int32),
                          -torch.ones(pts.shape[0], device=pts.device, dtype=torch.int32))
        winding = winding + torch.where(hit, sign, torch.zeros_like(sign))

    return winding


def _winding_cubic(p0, p1, p2, p3, pts):
    """Winding number contribution from a cubic Bezier for all query points.

    The curve is (-p0+3p1-3p2+p3)t^3 + (3p0-6p1+3p2)t^2 + (-3p0+3p1)t + p0
    """
    a_coeff = -p0[1] + 3*p1[1] - 3*p2[1] + p3[1]
    b_coeff = 3*p0[1] - 6*p1[1] + 3*p2[1]
    c_coeff = -3*p0[1] + 3*p1[1]
    d_coeff = p0[1]

    winding = torch.zeros(pts.shape[0], device=pts.device, dtype=torch.int32)

    d_val = d_coeff - pts[:, 1]
    roots, num_r = solve_cubic(
        a_coeff.expand(pts.shape[0]),
        b_coeff.expand(pts.shape[0]),
        c_coeff.expand(pts.shape[0]),
        d_val
    )

    ax = -p0[0] + 3*p1[0] - 3*p2[0] + p3[0]
    bx = 3*p0[0] - 6*p1[0] + 3*p2[0]
    cx = -3*p0[0] + 3*p1[0]

    for j in range(3):
        t = roots[:, j].to(pts.dtype)
        v = (j < num_r) & (t >= 0) & (t <= 1)
        tp = ax * t*t*t + bx * t*t + cx * t + p0[0] - pts[:, 0]
        hit = v & (tp > 0)
        # Derivative: 3*a*t^2 + 2*b*t + c
        deriv = 3*a_coeff * t*t + 2*b_coeff * t + c_coeff
        sign = torch.where(deriv > 0,
                          torch.ones(pts.shape[0], device=pts.device, dtype=torch.int32),
                          -torch.ones(pts.shape[0], device=pts.device, dtype=torch.int32))
        winding = winding + torch.where(hit, sign, torch.zeros_like(sign))

    return winding


def compute_winding_number(shape_type, shape_data, pts):
    """Compute winding number for a single shape at all query points.

    Args:
        shape_type: ShapeType enum
        shape_data: dict with shape parameters
        pts: [N, 2] query points (in local space)

    Returns: [N] integer winding numbers
    """
    if shape_type == ShapeType.circle:
        center = shape_data['center']
        radius = shape_data['radius']
        dist_sq = ((pts - center.unsqueeze(0)) ** 2).sum(dim=-1)
        return (dist_sq < radius * radius).to(torch.int32)

    elif shape_type == ShapeType.ellipse:
        center = shape_data['center']
        radius = shape_data['radius']
        diff = pts - center.unsqueeze(0)
        val = (diff[:, 0] / radius[0]) ** 2 + (diff[:, 1] / radius[1]) ** 2
        return (val < 1.0).to(torch.int32)

    elif shape_type == ShapeType.path:
        points = shape_data['points']
        ncp = shape_data['num_control_points']
        is_closed = shape_data['is_closed']
        lines, quads, cubics = extract_path_segments(points, ncp, is_closed)

        winding = torch.zeros(pts.shape[0], device=pts.device, dtype=torch.int32)
        for (i0, i1) in lines:
            winding = winding + _winding_line(points[i0], points[i1], pts)
        for (i0, i1, i2) in quads:
            winding = winding + _winding_quadratic(
                points[i0], points[i1], points[i2], pts)
        for (i0, i1, i2, i3) in cubics:
            winding = winding + _winding_cubic(
                points[i0], points[i1], points[i2], points[i3], pts)
        return winding

    elif shape_type == ShapeType.rect:
        p_min = shape_data['p_min']
        p_max = shape_data['p_max']
        inside = ((pts[:, 0] > p_min[0]) & (pts[:, 0] < p_max[0]) &
                  (pts[:, 1] > p_min[1]) & (pts[:, 1] < p_max[1]))
        return inside.to(torch.int32)


# ============================================================================
# Distance Computation (differentiable)
# ============================================================================

def _closest_point_line(p0, p1, pts):
    """Closest point on a line segment to each query point. Differentiable.

    Returns: closest_pts [N, 2], distances [N], t_values [N]
    """
    d = p1 - p0
    denom = (d * d).sum()
    if denom < 1e-10:
        cp = p0.unsqueeze(0).expand(pts.shape[0], 2)
        return cp, torch.sqrt(((pts - cp) ** 2).sum(dim=-1)), torch.zeros(pts.shape[0], device=pts.device)

    t = ((pts - p0.unsqueeze(0)) * d.unsqueeze(0)).sum(dim=-1) / denom
    t_clamped = t.clamp(0.0, 1.0)
    cp = p0.unsqueeze(0) + t_clamped.unsqueeze(-1) * d.unsqueeze(0)
    dist = torch.sqrt(((pts - cp) ** 2).sum(dim=-1) + 1e-20)
    return cp, dist, t_clamped


def _closest_point_quadratic(p0, p1, p2, pts):
    """Closest point on a quadratic Bezier to each query point.

    Uses envelope theorem: find t with no_grad, then evaluate differentiably.
    """
    N = pts.shape[0]
    device = pts.device

    # Find closest t using cubic solver (no grad needed for t)
    with torch.no_grad():
        # (q - pt) · q' = 0 where q = (p0-2p1+p2)t^2 + (-2p0+2p1)t + p0
        A = ((p0-2*p1+p2)**2).sum()
        B = (3*(p0-2*p1+p2)*(-p0+p1)).sum()
        C_base = (2*(-p0+p1)**2).sum() + ((p0-2*p1+p2)*p0).sum()
        D_base = ((-p0+p1)*p0).sum()

        # Per-point terms
        C_pt = -((p0-2*p1+p2) * pts).sum(dim=-1)
        D_pt = -((-p0+p1) * pts).sum(dim=-1)

        C_full = C_base + C_pt
        D_full = D_base + D_pt

        roots, num_r = solve_cubic(
            A.expand(N), B.expand(N), C_full, D_full)

        # Also check endpoints
        best_t = torch.zeros(N, device=device)
        best_dist = torch.full((N,), float('inf'), device=device)

        for endpoint_t in [0.0, 1.0]:
            tt = 1 - endpoint_t
            cp = tt*tt*p0 + 2*tt*endpoint_t*p1 + endpoint_t**2*p2
            d = torch.sqrt(((pts - cp.unsqueeze(0))**2).sum(dim=-1) + 1e-20)
            better = d < best_dist
            best_dist = torch.where(better, d, best_dist)
            best_t = torch.where(better, torch.full_like(best_t, endpoint_t), best_t)

        for j in range(3):
            t = roots[:, j].to(pts.dtype).clamp(0.0, 1.0)
            v = j < num_r
            tt = 1 - t
            cp = tt.unsqueeze(-1)**2 * p0.unsqueeze(0) + \
                 (2*tt*t).unsqueeze(-1) * p1.unsqueeze(0) + \
                 t.unsqueeze(-1)**2 * p2.unsqueeze(0)
            d = torch.sqrt(((pts - cp)**2).sum(dim=-1) + 1e-20)
            better = v & (d < best_dist)
            best_dist = torch.where(better, d, best_dist)
            best_t = torch.where(better, t, best_t)

    # Evaluate at best_t differentiably (envelope theorem)
    t = best_t.detach()
    tt = 1 - t
    cp = tt.unsqueeze(-1)**2 * p0.unsqueeze(0) + \
         (2*tt*t).unsqueeze(-1) * p1.unsqueeze(0) + \
         t.unsqueeze(-1)**2 * p2.unsqueeze(0)
    dist = torch.sqrt(((pts - cp)**2).sum(dim=-1) + 1e-20)
    return cp, dist, t


def _closest_point_cubic(p0, p1, p2, p3, pts):
    """Closest point on a cubic Bezier to each query point.

    Uses quintic polynomial root isolation + Newton/bisection, then envelope theorem.
    For efficiency, we use a simpler approach: sample + refine.
    """
    N = pts.shape[0]
    device = pts.device

    with torch.no_grad():
        # Sample the curve at multiple points and find approximate closest
        n_samples = 16
        t_samples = torch.linspace(0, 1, n_samples, device=device)
        tt_samples = 1 - t_samples

        # Evaluate curve at all sample points: [n_samples, 2]
        curve_pts = (tt_samples**3).unsqueeze(-1) * p0.unsqueeze(0) + \
                    (3*tt_samples**2*t_samples).unsqueeze(-1) * p1.unsqueeze(0) + \
                    (3*tt_samples*t_samples**2).unsqueeze(-1) * p2.unsqueeze(0) + \
                    (t_samples**3).unsqueeze(-1) * p3.unsqueeze(0)

        # Distance from each query point to each sample: [N, n_samples]
        dists = torch.sqrt(((pts.unsqueeze(1) - curve_pts.unsqueeze(0))**2).sum(dim=-1) + 1e-20)
        best_idx = dists.argmin(dim=1)
        best_t = t_samples[best_idx]

        # Refine with Newton's method
        for _ in range(8):
            t = best_t
            tt = 1 - t

            # q(t) = curve point
            q = tt.unsqueeze(-1)**3 * p0 + \
                (3*tt**2*t).unsqueeze(-1) * p1 + \
                (3*tt*t**2).unsqueeze(-1) * p2 + \
                t.unsqueeze(-1)**3 * p3

            # q'(t) = derivative
            qp = (3*tt**2).unsqueeze(-1) * (p1-p0) + \
                 (6*tt*t).unsqueeze(-1) * (p2-p1) + \
                 (3*t**2).unsqueeze(-1) * (p3-p2)

            # q''(t) = second derivative
            qpp = (6*tt).unsqueeze(-1) * (p2 - 2*p1 + p0) + \
                  (6*t).unsqueeze(-1) * (p3 - 2*p2 + p1)

            diff = q - pts
            # f(t) = (q - pt) · q' = 0
            f = (diff * qp).sum(dim=-1)
            # f'(t) = q' · q' + (q - pt) · q''
            fp = (qp * qp).sum(dim=-1) + (diff * qpp).sum(dim=-1)

            safe_fp = torch.where(fp.abs() > 1e-10, fp, torch.ones_like(fp))
            dt = -f / safe_fp
            best_t = (best_t + dt).clamp(0.0, 1.0)

    # Evaluate differentiably at best_t
    t = best_t.detach()
    tt = 1 - t
    cp = tt.unsqueeze(-1)**3 * p0.unsqueeze(0) + \
         (3*tt**2*t).unsqueeze(-1) * p1.unsqueeze(0) + \
         (3*tt*t**2).unsqueeze(-1) * p2.unsqueeze(0) + \
         t.unsqueeze(-1)**3 * p3.unsqueeze(0)
    dist = torch.sqrt(((pts - cp)**2).sum(dim=-1) + 1e-20)
    return cp, dist, t


def compute_distance(shape_type, shape_data, pts, stroke_width=0.0):
    """Compute distance from query points to nearest shape boundary.

    Returns:
        closest_pts: [N, 2]
        distances: [N] (unsigned)
        found: [N] bool
    """
    N = pts.shape[0]
    device = pts.device

    if shape_type == ShapeType.circle:
        center = shape_data['center']
        radius = shape_data['radius']
        diff = pts - center.unsqueeze(0)
        dist_to_center = torch.sqrt((diff**2).sum(dim=-1) + 1e-20)
        safe_dist = torch.where(dist_to_center > 1e-10, dist_to_center,
                               torch.ones_like(dist_to_center))
        cp = center.unsqueeze(0) + radius * diff / safe_dist.unsqueeze(-1)
        dist = (dist_to_center - radius).abs()
        return cp, dist, torch.ones(N, device=device, dtype=torch.bool)

    elif shape_type == ShapeType.ellipse:
        # Approximate: treat as circle with average radius
        center = shape_data['center']
        radius = shape_data['radius']
        diff = pts - center.unsqueeze(0)
        # Normalize to unit circle space
        norm_diff = diff / radius.unsqueeze(0)
        norm_dist = torch.sqrt((norm_diff**2).sum(dim=-1) + 1e-20)
        safe_nd = torch.where(norm_dist > 1e-10, norm_dist, torch.ones_like(norm_dist))
        cp = center.unsqueeze(0) + radius.unsqueeze(0) * norm_diff / safe_nd.unsqueeze(-1)
        dist = torch.sqrt(((pts - cp)**2).sum(dim=-1) + 1e-20)
        return cp, dist, torch.ones(N, device=device, dtype=torch.bool)

    elif shape_type == ShapeType.path:
        points = shape_data['points']
        ncp = shape_data['num_control_points']
        is_closed = shape_data['is_closed']
        lines, quads, cubics = extract_path_segments(points, ncp, is_closed)

        best_dist = torch.full((N,), float('inf'), device=device)
        best_cp = torch.zeros(N, 2, device=device)
        found = torch.zeros(N, device=device, dtype=torch.bool)

        for (i0, i1) in lines:
            cp, dist, _ = _closest_point_line(points[i0], points[i1], pts)
            better = dist < best_dist
            best_dist = torch.where(better, dist, best_dist)
            best_cp = torch.where(better.unsqueeze(-1), cp, best_cp)
            found = found | better

        for (i0, i1, i2) in quads:
            cp, dist, _ = _closest_point_quadratic(
                points[i0], points[i1], points[i2], pts)
            better = dist < best_dist
            best_dist = torch.where(better, dist, best_dist)
            best_cp = torch.where(better.unsqueeze(-1), cp, best_cp)
            found = found | better

        for (i0, i1, i2, i3) in cubics:
            cp, dist, _ = _closest_point_cubic(
                points[i0], points[i1], points[i2], points[i3], pts)
            better = dist < best_dist
            best_dist = torch.where(better, dist, best_dist)
            best_cp = torch.where(better.unsqueeze(-1), cp, best_cp)
            found = found | better

        return best_cp, best_dist, found

    elif shape_type == ShapeType.rect:
        p_min = shape_data['p_min']
        p_max = shape_data['p_max']
        # Distance to each edge
        edges = [
            (p_min, torch.stack([p_min[0], p_max[1]])),  # left
            (p_min, torch.stack([p_max[0], p_min[1]])),  # top
            (torch.stack([p_max[0], p_min[1]]), p_max),  # right
            (torch.stack([p_min[0], p_max[1]]), p_max),  # bottom
        ]
        best_dist = torch.full((N,), float('inf'), device=device)
        best_cp = torch.zeros(N, 2, device=device)
        for e0, e1 in edges:
            cp, dist, _ = _closest_point_line(e0, e1, pts)
            better = dist < best_dist
            best_dist = torch.where(better, dist, best_dist)
            best_cp = torch.where(better.unsqueeze(-1), cp, best_cp)
        return best_cp, best_dist, torch.ones(N, device=device, dtype=torch.bool)


# ============================================================================
# Within-Distance Test (for strokes)
# ============================================================================

def within_distance_shape(shape_type, shape_data, pts, stroke_width):
    """Test if query points are within stroke_width of a shape boundary.

    Returns: [N] boolean mask
    """
    if shape_type == ShapeType.circle:
        center = shape_data['center']
        radius = shape_data['radius']
        dist = torch.sqrt(((pts - center.unsqueeze(0))**2).sum(dim=-1) + 1e-20)
        return (dist - radius).abs() < stroke_width

    elif shape_type == ShapeType.ellipse:
        center = shape_data['center']
        radius = shape_data['radius']
        norm_diff = (pts - center.unsqueeze(0)) / radius.unsqueeze(0)
        norm_dist = torch.sqrt((norm_diff**2).sum(dim=-1) + 1e-20)
        # Approximate distance
        avg_r = (radius[0] + radius[1]) / 2
        return (norm_dist * avg_r - avg_r).abs() < stroke_width

    elif shape_type == ShapeType.path:
        _, dist, found = compute_distance(shape_type, shape_data, pts)
        return found & (dist < stroke_width)

    elif shape_type == ShapeType.rect:
        _, dist, _ = compute_distance(shape_type, shape_data, pts)
        return dist < stroke_width


# ============================================================================
# Color Sampling (differentiable)
# ============================================================================

def sample_color(color_type, color_data, pts):
    """Sample color at query points. Differentiable w.r.t. color parameters.

    Args:
        color_type: ColorType enum or None
        color_data: dict or tensor with color parameters
        pts: [N, 2] query points (canvas space)

    Returns: [N, 4] RGBA colors
    """
    if color_type is None:
        return None

    N = pts.shape[0]
    device = pts.device

    if color_type == ColorType.constant:
        color = color_data  # [4] tensor
        return color.unsqueeze(0).expand(N, 4)

    elif color_type == ColorType.linear_gradient:
        begin = color_data['begin']  # [2]
        end = color_data['end']      # [2]
        offsets = color_data['offsets']  # [S]
        stop_colors = color_data['stop_colors']  # [S, 4]

        # Project pts onto gradient line
        d = end - begin
        denom = (d * d).sum().clamp(min=1e-3)
        t = ((pts - begin.unsqueeze(0)) * d.unsqueeze(0)).sum(dim=-1) / denom

        return _interpolate_stops(t, offsets, stop_colors)

    elif color_type == ColorType.radial_gradient:
        center = color_data['center']  # [2]
        radius = color_data['radius']  # [2]
        offsets = color_data['offsets']  # [S]
        stop_colors = color_data['stop_colors']  # [S, 4]

        offset = pts - center.unsqueeze(0)
        normalized = offset / radius.unsqueeze(0)
        t = torch.sqrt((normalized**2).sum(dim=-1) + 1e-20)

        return _interpolate_stops(t, offsets, stop_colors)


def _interpolate_stops(t, offsets, stop_colors):
    """Interpolate color at parameter t using gradient stops. Differentiable.

    Args:
        t: [N] parameter values
        offsets: [S] stop offsets (sorted)
        stop_colors: [S, 4] stop colors

    Returns: [N, 4] interpolated colors
    """
    N = t.shape[0]
    S = offsets.shape[0]
    device = t.device

    if S == 1:
        return stop_colors[0:1].expand(N, 4)

    # Clamp t to [offsets[0], offsets[-1]]
    t_clamped = t.clamp(offsets[0], offsets[-1])

    # Find which interval each t falls in using searchsorted
    # searchsorted returns index where t would be inserted
    idx = torch.searchsorted(offsets, t_clamped).clamp(1, S - 1)

    # Interpolation within the interval
    o_lo = offsets[idx - 1]
    o_hi = offsets[idx]
    c_lo = stop_colors[idx - 1]
    c_hi = stop_colors[idx]

    span = (o_hi - o_lo).clamp(min=1e-6)
    frac = ((t_clamped - o_lo) / span).clamp(0, 1)

    return c_lo * (1 - frac).unsqueeze(-1) + c_hi * frac.unsqueeze(-1)


# ============================================================================
# Filter Weight Computation
# ============================================================================

def compute_filter_weight(filter_type, radius, dx, dy):
    """Compute filter weight. dx, dy are offsets from pixel center.

    All args can be tensors for vectorized computation.
    """
    outside = (dx.abs() > radius) | (dy.abs() > radius)

    if filter_type == FilterType.box:
        w = torch.ones_like(dx) / (2 * radius) ** 2
    elif filter_type == FilterType.tent:
        w = (radius - dx.abs()) * (radius - dy.abs()) / radius ** 4
    elif filter_type == FilterType.parabolic: # radial parabolic
        w = (4.0/3.0) * (1 - (dx/radius)**2) * (4.0/3.0) * (1 - (dy/radius)**2)
    elif filter_type == FilterType.hann: # Hann
        ndx = dx / (2*radius) + 0.5
        ndy = dy / (2*radius) + 0.5
        w = (0.5 * (1 - torch.cos(2*math.pi*ndx)) *
             0.5 * (1 - torch.cos(2*math.pi*ndy)) / radius**2)
    else:
        w = torch.ones_like(dx) / (2 * radius) ** 2

    return torch.where(outside, torch.zeros_like(w), w)


# ============================================================================
# Smoothstep (for prefiltered rendering)
# ============================================================================

def smoothstep(d):
    t = ((d + 1.0) / 2.0).clamp(0.0, 1.0)
    return t * t * (3 - 2 * t)


# ============================================================================
# Transform utilities
# ============================================================================

def xform_pt(matrix, pts):
    """Transform points by a 3x3 matrix (homogeneous).

    Args:
        matrix: [3, 3] transformation matrix
        pts: [N, 2] points

    Returns: [N, 2] transformed points
    """
    ones = torch.ones(pts.shape[0], 1, device=pts.device, dtype=pts.dtype)
    homo = torch.cat([pts, ones], dim=-1)  # [N, 3]
    result = (homo @ matrix.T)  # [N, 3]
    w = result[:, 2:3].clamp(min=1e-10)
    return result[:, :2] / w


# ============================================================================
# Boundary Sampling (for Reynolds transport theorem in backward pass)
# ============================================================================

def _path_segment_lengths(points, num_control_points, is_closed):
    """Compute approximate lengths of each path segment."""
    lines, quads, cubics = extract_path_segments(points, num_control_points, is_closed)
    n_segs = num_control_points.shape[0]
    lengths = torch.zeros(n_segs, device=points.device)

    seg_idx = 0
    point_id = 0
    for i in range(n_segs):
        ncp = num_control_points[i].item()
        n_pts_seg = ncp + 2
        # Approximate length by sampling
        t_vals = torch.linspace(0, 1, 10, device=points.device)

        if ncp == 0:
            i0 = point_id
            i1 = (point_id + 1) % points.shape[0]
            lengths[i] = torch.sqrt(((points[i1] - points[i0])**2).sum() + 1e-20)
        elif ncp == 1:
            i0 = point_id
            i1 = point_id + 1
            i2 = (point_id + 2) % points.shape[0]
            tt = 1 - t_vals
            curve = tt.unsqueeze(-1)**2 * points[i0] + \
                    (2*tt*t_vals).unsqueeze(-1) * points[i1] + \
                    t_vals.unsqueeze(-1)**2 * points[i2]
            diffs = curve[1:] - curve[:-1]
            lengths[i] = torch.sqrt((diffs**2).sum(dim=-1) + 1e-20).sum()
        elif ncp == 2:
            i0 = point_id
            i1 = point_id + 1
            i2 = point_id + 2
            i3 = (point_id + 3) % points.shape[0]
            tt = 1 - t_vals
            curve = tt.unsqueeze(-1)**3 * points[i0] + \
                    (3*tt**2*t_vals).unsqueeze(-1) * points[i1] + \
                    (3*tt*t_vals**2).unsqueeze(-1) * points[i2] + \
                    t_vals.unsqueeze(-1)**3 * points[i3]
            diffs = curve[1:] - curve[:-1]
            lengths[i] = torch.sqrt((diffs**2).sum(dim=-1) + 1e-20).sum()
        point_id += ncp + 1
    return lengths


def sample_boundary_point(shape_type, shape_data, t_val, stroke_width=0.0,
                          stroke_perturb_direction=0.0):
    """Sample a point on the boundary of a shape.

    Args:
        shape_type: ShapeType enum
        shape_data: dict with shape parameters
        t_val: float in [0, 1) - random parameter
        stroke_width: float
        stroke_perturb_direction: -1, 0, or 1

    Returns:
        boundary_pt: [2] tensor
        normal: [2] tensor
        pdf: float tensor
    """
    device = next(iter(shape_data.values())).device if isinstance(shape_data, dict) else shape_data.device

    if shape_type == ShapeType.circle:
        center = shape_data['center']
        radius = shape_data['radius']
        angle = 2 * math.pi * t_val
        offset = torch.stack([radius * math.cos(angle),
                              radius * math.sin(angle)])
        normal = offset / torch.sqrt((offset**2).sum() + 1e-20)
        pdf = torch.tensor(1.0 / (2 * math.pi * radius.item()), device=device)
        pt = center + offset
        if stroke_perturb_direction != 0:
            pt = pt + stroke_perturb_direction * stroke_width * normal
            if stroke_perturb_direction < 0:
                normal = -normal
        return pt, normal, pdf

    elif shape_type == ShapeType.ellipse:
        center = shape_data['center']
        r = shape_data['radius']
        angle = 2 * math.pi * t_val
        offset = torch.stack([r[0] * math.cos(angle),
                              r[1] * math.sin(angle)])
        dxdt = -r[0] * math.sin(angle) * 2 * math.pi
        dydt = r[1] * math.cos(angle) * 2 * math.pi
        tangent_len = math.sqrt(dxdt**2 + dydt**2)
        normal = torch.tensor([dydt, -dxdt], device=device) / (tangent_len + 1e-10)
        pdf = torch.tensor(1.0 / (tangent_len + 1e-10), device=device)
        pt = center + offset
        if stroke_perturb_direction != 0:
            pt = pt + stroke_perturb_direction * stroke_width * normal
            if stroke_perturb_direction < 0:
                normal = -normal
        return pt, normal, pdf

    elif shape_type == ShapeType.path:
        points = shape_data['points']
        ncp = shape_data['num_control_points']
        is_closed = shape_data['is_closed']

        lengths = _path_segment_lengths(points, ncp, is_closed)
        total_length = lengths.sum()
        if total_length < 1e-10:
            return torch.zeros(2, device=device), torch.zeros(2, device=device), torch.tensor(0.0, device=device)

        # CDF for sampling segments
        cdf = torch.cumsum(lengths / total_length, dim=0)

        # Find which segment to sample
        seg_idx = torch.searchsorted(cdf, torch.tensor(t_val, device=device)).clamp(0, ncp.shape[0]-1).item()
        # Local t within segment
        prev_cdf = cdf[seg_idx - 1].item() if seg_idx > 0 else 0.0
        seg_cdf = cdf[seg_idx].item()
        local_t = (t_val - prev_cdf) / max(seg_cdf - prev_cdf, 1e-10)
        local_t = max(0.0, min(1.0, local_t))

        # Get segment point indices
        point_id = 0
        for i in range(seg_idx):
            point_id += ncp[i].item() + 1
        n_ctrl = ncp[seg_idx].item()
        n_pts = points.shape[0]

        seg_pmf = (lengths[seg_idx] / total_length).item()

        if n_ctrl == 0:
            i0 = point_id
            i1 = (point_id + 1) % n_pts
            p0 = points[i0]
            p1 = points[i1]
            tangent = p1 - p0
            tan_len = torch.sqrt((tangent**2).sum() + 1e-20)
            normal = torch.stack([-tangent[1], tangent[0]]) / tan_len
            pdf = torch.tensor(seg_pmf, device=device) / tan_len
            pt = p0 + local_t * (p1 - p0)
        elif n_ctrl == 1:
            i0 = point_id
            i1 = point_id + 1
            i2 = (point_id + 2) % n_pts
            p0 = points[i0]
            p1 = points[i1]
            p2 = points[i2]
            t = local_t
            tt = 1 - t
            pt = tt**2 * p0 + 2*tt*t * p1 + t**2 * p2
            tangent = 2*tt*(p1 - p0) + 2*t*(p2 - p1)
            tan_len = torch.sqrt((tangent**2).sum() + 1e-20)
            normal = torch.stack([-tangent[1], tangent[0]]) / tan_len
            pdf = torch.tensor(seg_pmf, device=device) / tan_len
        elif n_ctrl == 2:
            i0 = point_id
            i1 = point_id + 1
            i2 = point_id + 2
            i3 = (point_id + 3) % n_pts
            p0 = points[i0]
            p1 = points[i1]
            p2 = points[i2]
            p3 = points[i3]
            t = local_t
            tt = 1 - t
            pt = tt**3*p0 + 3*tt**2*t*p1 + 3*tt*t**2*p2 + t**3*p3
            tangent = 3*tt**2*(p1-p0) + 6*tt*t*(p2-p1) + 3*t**2*(p3-p2)
            tan_len = torch.sqrt((tangent**2).sum() + 1e-20)
            normal = torch.stack([-tangent[1], tangent[0]]) / tan_len
            pdf = torch.tensor(seg_pmf, device=device) / tan_len
        else:
            return torch.zeros(2, device=device), torch.zeros(2, device=device), torch.tensor(0.0, device=device)

        if stroke_perturb_direction != 0:
            pt = pt + stroke_perturb_direction * stroke_width * normal
            if stroke_perturb_direction < 0:
                normal = -normal

        return pt, normal, pdf

    elif shape_type == ShapeType.rect:
        p_min = shape_data['p_min']
        p_max = shape_data['p_max']
        w = p_max[0] - p_min[0]
        h = p_max[1] - p_min[1]
        perimeter = 2 * (w + h)
        pdf = torch.tensor(1.0 / perimeter.item(), device=device)

        pos = t_val * perimeter.item()
        if pos < w.item():
            pt = torch.stack([p_min[0] + pos, p_min[1]])
            normal = torch.tensor([0.0, -1.0], device=device)
        elif pos < (w + h).item():
            pt = torch.stack([p_max[0], p_min[1] + (pos - w.item())])
            normal = torch.tensor([1.0, 0.0], device=device)
        elif pos < (2*w + h).item():
            pt = torch.stack([p_max[0] - (pos - w.item() - h.item()), p_max[1]])
            normal = torch.tensor([0.0, 1.0], device=device)
        else:
            pt = torch.stack([p_min[0], p_max[1] - (pos - 2*w.item() - h.item())])
            normal = torch.tensor([-1.0, 0.0], device=device)

        if stroke_perturb_direction != 0:
            pt = pt + stroke_perturb_direction * stroke_width * normal
            if stroke_perturb_direction < 0:
                normal = -normal

        return pt, normal, pdf


# ============================================================================
# Main Rendering Function
# ============================================================================

def render_scene(width, height, num_samples_x, num_samples_y, seed,
                 canvas_width, canvas_height,
                 shapes, shape_groups,
                 filter_type, filter_radius,
                 background_image=None,
                 use_prefiltering=False,
                 device=None):
    """Render a vector graphics scene. Forward pass only.

    This builds a computation graph suitable for autograd differentiation
    when use_prefiltering=True. For use_prefiltering=False, use
    DiffVGFunction which adds boundary sampling gradients.

    Args:
        width, height: output image dimensions
        num_samples_x, num_samples_y: samples per pixel in each dimension
        seed: random seed
        canvas_width, canvas_height: canvas dimensions
        shapes: list of (shape_type, shape_data) tuples
        shape_groups: list of shape group dicts
        filter_type: FilterType enum
        filter_radius: float
        background_image: optional [H, W, 4] tensor
        use_prefiltering: if True, use soft boundaries (fully differentiable)
        device: torch device

    Returns:
        rendered_image: [H, W, 4] tensor
    """
    if device is None:
        device = torch.device('cpu')

    num_samples = num_samples_x * num_samples_y

    # Generate sample positions with jittered sampling
    gen = torch.Generator(device='cpu').manual_seed(seed)

    # Create grid of pixel positions
    py_coords = torch.arange(height, device=device, dtype=torch.float32)
    px_coords = torch.arange(width, device=device, dtype=torch.float32)
    sy_coords = torch.arange(num_samples_y, device=device, dtype=torch.float32)
    sx_coords = torch.arange(num_samples_x, device=device, dtype=torch.float32)

    # [H, W, sy, sx] grid
    grid_y, grid_x, grid_sy, grid_sx = torch.meshgrid(
        py_coords, px_coords, sy_coords, sx_coords, indexing='ij')

    # Random jitter
    N_total = height * width * num_samples_y * num_samples_x
    jitter = torch.rand(N_total, 2, generator=gen, device='cpu').to(device)
    if use_prefiltering:
        jitter = torch.full_like(jitter, 0.5)

    rx = jitter[:, 0].reshape(height, width, num_samples_y, num_samples_x)
    ry = jitter[:, 1].reshape(height, width, num_samples_y, num_samples_x)

    # Sample positions in pixel space
    sample_x = grid_x + (grid_sx + rx) / num_samples_x
    sample_y = grid_y + (grid_sy + ry) / num_samples_y

    # Flatten to [N, 2]
    pts_pixel = torch.stack([sample_x.reshape(-1), sample_y.reshape(-1)], dim=-1)
    N = pts_pixel.shape[0]

    # Normalize to [0, 1)
    pts_norm = pts_pixel.clone()
    pts_norm[:, 0] /= width
    pts_norm[:, 1] /= height

    # Canvas space
    pts_canvas = pts_pixel.clone()
    pts_canvas[:, 0] *= canvas_width / width
    pts_canvas[:, 1] *= canvas_height / height

    # ---- Compute filter weights ----
    weight_image = torch.zeros(height, width, device=device)
    px_int = pts_pixel[:, 0].long().reshape(height, width, num_samples_y, num_samples_x)
    py_int = pts_pixel[:, 1].long().reshape(height, width, num_samples_y, num_samples_x)

    ri = int(math.ceil(filter_radius))
    for dy in range(-ri, ri + 1):
        for dx in range(-ri, ri + 1):
            xx = px_int + dx
            yy = py_int + dy
            valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            xc = xx.float() + 0.5
            yc = yy.float() + 0.5
            fw = compute_filter_weight(
                filter_type, filter_radius,
                xc - sample_x, yc - sample_y)
            fw = torch.where(valid, fw, torch.zeros_like(fw))
            # Scatter add to weight image
            flat_idx = yy.clamp(0, height-1) * width + xx.clamp(0, width-1)
            weight_image.view(-1).scatter_add_(0, flat_idx.reshape(-1),
                                               fw.reshape(-1))

    # ---- Render each sample ----
    # For each sample, compute the color via fragment blending
    # We accumulate directly into the output image

    render_image = torch.zeros(height, width, 4, device=device)

    # Process all samples in a batch
    # For each shape group, check fill/stroke hits

    # First pass: for all samples, determine fragments
    # A fragment is (group_id, color_rgba, is_stroke)
    # We need to blend them per-sample

    # Strategy: compute per-sample color by iterating shape groups in order
    # (back to front), blending as we go.

    # Initialize per-sample accumulated color
    if background_image is not None:
        bg = background_image.to(device)
        # Lookup background at each sample's pixel
        px_idx = pts_pixel[:, 0].long().clamp(0, width-1)
        py_idx = pts_pixel[:, 1].long().clamp(0, height-1)
        sample_bg = bg[py_idx, px_idx]  # [N, 4]
        accum_color = sample_bg[:, :3].clone()
        accum_alpha = sample_bg[:, 3].clone()
    else:
        accum_color = torch.zeros(N, 3, device=device)
        accum_alpha = torch.zeros(N, device=device)

    # Process shape groups front to back doesn't work for alpha blending.
    # We need back to front (increasing group_id).
    # Shape groups are ordered by their index (0 = back, N-1 = front)

    for group_idx, sg in enumerate(shape_groups):
        sg_shape_ids = sg['shape_ids']
        fill_color_type = sg.get('fill_color_type')
        fill_color_data = sg.get('fill_color_data')
        stroke_color_type = sg.get('stroke_color_type')
        stroke_color_data = sg.get('stroke_color_data')
        use_even_odd = sg.get('use_even_odd_rule', True)
        shape_to_canvas = sg.get('shape_to_canvas')  # [3, 3]

        # Compute canvas_to_shape
        canvas_to_shape = torch.linalg.inv(shape_to_canvas)

        # Transform query points to local space
        local_pts = xform_pt(canvas_to_shape, pts_canvas)

        # Check fill and stroke for each shape in the group
        fill_mask = torch.zeros(N, device=device, dtype=torch.bool)
        stroke_mask = torch.zeros(N, device=device, dtype=torch.bool)
        stroke_distance = torch.full((N,), float('inf'), device=device)
        stroke_closest = torch.zeros(N, 2, device=device)

        for shape_id in sg_shape_ids:
            sid = shape_id.item() if isinstance(shape_id, torch.Tensor) else shape_id
            shape_type, shape_data = shapes[sid]
            stroke_w = shape_data.get('stroke_width', torch.tensor(0.0))
            if isinstance(stroke_w, (int, float)):
                stroke_w = torch.tensor(float(stroke_w), device=device)

            if fill_color_type is not None:
                wn = compute_winding_number(shape_type, shape_data, local_pts)
                if use_even_odd:
                    fill_mask = fill_mask | (wn.abs() % 2 == 1)
                else:
                    fill_mask = fill_mask | (wn != 0)

            if stroke_color_type is not None and stroke_w > 0:
                mask = within_distance_shape(shape_type, shape_data,
                                            local_pts, stroke_w)
                stroke_mask = stroke_mask | mask

                if use_prefiltering:
                    cp, dist, found = compute_distance(
                        shape_type, shape_data, local_pts)
                    better = found & (dist < stroke_distance)
                    stroke_distance = torch.where(better, dist, stroke_distance)
                    stroke_closest = torch.where(better.unsqueeze(-1), cp, stroke_closest)

        # Sample colors at canvas-space points
        def _blend(accum_color, accum_alpha, new_color, new_alpha, mask):
            """Alpha-blend new fragments onto accumulated color."""
            nc = new_color  # [N, 3]
            na = new_alpha  # [N]
            # Only blend where mask is True
            na = torch.where(mask, na, torch.zeros_like(na))
            nc = torch.where(mask.unsqueeze(-1), nc, torch.zeros_like(nc))
            out_color = accum_color * (1 - na.unsqueeze(-1)) + na.unsqueeze(-1) * nc
            out_alpha = accum_alpha * (1 - na) + na
            return out_color, out_alpha

        # Fill
        if fill_color_type is not None:
            fill_rgba = sample_color(fill_color_type, fill_color_data, pts_canvas)
            if use_prefiltering:
                # Compute soft alpha using signed distance
                _, dist, found = compute_distance(
                    shapes[sg_shape_ids[0].item()][0],
                    shapes[sg_shape_ids[0].item()][1],
                    local_pts)
                # Get winding for sign
                wn = compute_winding_number(
                    shapes[sg_shape_ids[0].item()][0],
                    shapes[sg_shape_ids[0].item()][1],
                    local_pts)
                inside = (wn.abs() % 2 == 1) if use_even_odd else (wn != 0)
                signed_dist = torch.where(inside, dist, -dist)
                soft_alpha = smoothstep(signed_dist) * fill_rgba[:, 3]
                accum_color, accum_alpha = _blend(
                    accum_color, accum_alpha,
                    fill_rgba[:, :3], soft_alpha,
                    torch.ones(N, device=device, dtype=torch.bool))
            else:
                accum_color, accum_alpha = _blend(
                    accum_color, accum_alpha,
                    fill_rgba[:, :3], fill_rgba[:, 3],
                    fill_mask)

        # Stroke
        if stroke_color_type is not None:
            stroke_rgba = sample_color(stroke_color_type, stroke_color_data, pts_canvas)
            if stroke_rgba is not None:
                stroke_w = shapes[sg_shape_ids[0].item()][1].get(
                    'stroke_width', torch.tensor(0.0))
                if isinstance(stroke_w, (int, float)):
                    stroke_w = torch.tensor(float(stroke_w), device=device)

                if use_prefiltering and stroke_w > 0:
                    abs_d = stroke_distance
                    soft_alpha = (smoothstep(abs_d + stroke_w) -
                                  smoothstep(abs_d - stroke_w)) * stroke_rgba[:, 3]
                    accum_color, accum_alpha = _blend(
                        accum_color, accum_alpha,
                        stroke_rgba[:, :3], soft_alpha,
                        torch.ones(N, device=device, dtype=torch.bool))
                else:
                    accum_color, accum_alpha = _blend(
                        accum_color, accum_alpha,
                        stroke_rgba[:, :3], stroke_rgba[:, 3],
                        stroke_mask)

    # Normalize by alpha
    safe_alpha = accum_alpha.clamp(min=1e-6)
    final_color = accum_color / safe_alpha.unsqueeze(-1)
    final_color = torch.where(accum_alpha.unsqueeze(-1) > 1e-6,
                              final_color, torch.zeros_like(final_color))
    sample_rgba = torch.cat([final_color, accum_alpha.unsqueeze(-1)], dim=-1)

    # ---- Splat samples to image using filter ----
    sample_rgba = sample_rgba.reshape(height, width, num_samples_y, num_samples_x, 4)

    for dy in range(-ri, ri + 1):
        for dx in range(-ri, ri + 1):
            xx = px_int + dx
            yy = py_int + dy
            valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            xc = xx.float() + 0.5
            yc = yy.float() + 0.5
            fw = compute_filter_weight(
                filter_type, filter_radius,
                xc - sample_x, yc - sample_y)
            fw = torch.where(valid, fw, torch.zeros_like(fw))

            # Weight by filter / weight_sum
            ws = weight_image[yy.clamp(0, height-1), xx.clamp(0, width-1)]
            safe_ws = ws.clamp(min=1e-10)
            weighted = (fw / safe_ws).unsqueeze(-1) * sample_rgba  # [H, W, sy, sx, 4]
            weighted = torch.where(valid.unsqueeze(-1), weighted, torch.zeros_like(weighted))

            flat_idx = (yy.clamp(0, height-1) * width + xx.clamp(0, width-1))
            # Sum over samples within each pixel
            for c in range(4):
                wc = weighted[..., c].reshape(-1)
                render_image[:, :, c].view(-1).scatter_add_(
                    0, flat_idx.reshape(-1), wc)

    return render_image


# ============================================================================
# Boundary Gradient Computation (Reynolds Transport Theorem)
# ============================================================================

def compute_boundary_gradients(width, height, num_samples_x, num_samples_y, seed,
                               canvas_width, canvas_height,
                               shapes, shape_groups,
                               filter_type, filter_radius,
                               d_render_image, weight_image,
                               background_image=None,
                               device=None):
    """Compute boundary gradient contributions using Reynolds transport theorem.

    This function samples points on shape boundaries, evaluates the color
    difference across each boundary, and accumulates gradients into shape
    and color parameters.

    Returns dict of gradients keyed by (group_idx, param_name).
    """
    if device is None:
        device = torch.device('cpu')

    num_samples = width * height * num_samples_x * num_samples_y
    gen = torch.Generator(device='cpu').manual_seed(seed + 1000000)

    # Collect all shapes with their boundary lengths for importance sampling
    all_boundary_shapes = []
    for group_idx, sg in enumerate(shape_groups):
        for shape_id in sg['shape_ids']:
            sid = shape_id.item() if isinstance(shape_id, torch.Tensor) else shape_id
            shape_type, shape_data = shapes[sid]
            # Estimate boundary length
            if shape_type == ShapeType.circle:
                length = 2 * math.pi * shape_data['radius'].item()
            elif shape_type == ShapeType.ellipse:
                r = shape_data['radius']
                length = 2 * math.pi * math.sqrt((r[0].item()**2 + r[1].item()**2) / 2)
            elif shape_type == ShapeType.path:
                lengths = _path_segment_lengths(
                    shape_data['points'], shape_data['num_control_points'],
                    shape_data['is_closed'])
                length = lengths.sum().item()
            elif shape_type == ShapeType.rect:
                p_min = shape_data['p_min']
                p_max = shape_data['p_max']
                length = 2 * ((p_max[0]-p_min[0]).item() + (p_max[1]-p_min[1]).item())
            else:
                length = 0.0
            if length > 0:
                all_boundary_shapes.append((group_idx, sid, length))

    if not all_boundary_shapes:
        return {}

    total_length = sum(l for _, _, l in all_boundary_shapes)
    shape_pmfs = [l / total_length for _, _, l in all_boundary_shapes]

    # Accumulate gradients
    grads = {}

    for sample_idx in range(num_samples):
        u = torch.rand(1, generator=gen).item()
        t = torch.rand(1, generator=gen).item()

        # Select which shape to sample
        cumsum = 0.0
        selected = 0
        for i, pmf in enumerate(shape_pmfs):
            cumsum += pmf
            if u < cumsum:
                selected = i
                break

        group_idx, shape_id, _ = all_boundary_shapes[selected]
        shape_pmf = shape_pmfs[selected]
        sg = shape_groups[group_idx]
        shape_type, shape_data = shapes[shape_id]
        shape_to_canvas = sg['shape_to_canvas']
        canvas_to_shape = torch.linalg.inv(shape_to_canvas)
        stroke_w = shape_data.get('stroke_width', torch.tensor(0.0))
        if isinstance(stroke_w, (int, float)):
            stroke_w = torch.tensor(float(stroke_w), device=device)

        has_fill = sg.get('fill_color_type') is not None
        has_stroke = sg.get('stroke_color_type') is not None and stroke_w > 0

        # Decide fill vs stroke boundary
        stroke_perturb = 0.0
        pdf_factor = 1.0
        t_sample = t
        if has_fill and has_stroke:
            if t_sample < 0.5:
                t_sample = 2 * t_sample
                pdf_factor = 0.5
            else:
                stroke_perturb = 1.0
                t_sample = 2 * (t_sample - 0.5)
                pdf_factor = 0.5
        elif has_stroke:
            stroke_perturb = 1.0

        if stroke_perturb != 0:
            if t_sample < 0.5:
                stroke_perturb = -1.0
                t_sample = 2 * t_sample
                pdf_factor *= 0.5
            else:
                stroke_perturb = 1.0
                t_sample = 2 * (t_sample - 0.5)
                pdf_factor *= 0.5

        local_pt, normal, boundary_pdf = sample_boundary_point(
            shape_type, shape_data, t_sample, stroke_w.item(), stroke_perturb)

        if boundary_pdf.item() <= 0:
            continue

        # Transform to canvas space
        boundary_pt_canvas = xform_pt(shape_to_canvas, local_pt.unsqueeze(0)).squeeze(0)
        normal_canvas = xform_pt(
            canvas_to_shape.T,
            normal.unsqueeze(0)).squeeze(0)
        normal_canvas = normal_canvas / torch.sqrt((normal_canvas**2).sum() + 1e-20)

        # Normalize to screen space
        bx = (boundary_pt_canvas[0] / canvas_width * width).long().item()
        by = (boundary_pt_canvas[1] / canvas_height * height).long().item()
        if bx < 0 or bx >= width or by < 0 or by >= height:
            continue

        # Compute d_color from d_render_image
        screen_pt = boundary_pt_canvas.clone()
        screen_pt[0] = screen_pt[0] / canvas_width * width
        screen_pt[1] = screen_pt[1] / canvas_height * height
        d_color = _gather_d_color(filter_type, filter_radius,
                                  d_render_image, weight_image,
                                  width, height, screen_pt)
        d_color = d_color / (canvas_width * canvas_height)

        # Sample color on both sides
        eps = 1e-4
        inside_pt = boundary_pt_canvas - eps * normal_canvas
        outside_pt = boundary_pt_canvas + eps * normal_canvas

        inside_npt = inside_pt.unsqueeze(0)
        inside_npt = inside_npt.clone()
        inside_npt[:, 0] /= canvas_width
        inside_npt[:, 1] /= canvas_height

        outside_npt = outside_pt.unsqueeze(0)
        outside_npt = outside_npt.clone()
        outside_npt[:, 0] /= canvas_width
        outside_npt[:, 1] /= canvas_height

        # Simple color evaluation at the boundary (just sample from this group)
        color_inside = _eval_scene_color(
            inside_pt.unsqueeze(0), shapes, shape_groups, canvas_width, canvas_height,
            background_image, device)
        color_outside = _eval_scene_color(
            outside_pt.unsqueeze(0), shapes, shape_groups, canvas_width, canvas_height,
            background_image, device)

        color_diff = color_inside.squeeze(0) - color_outside.squeeze(0)
        pdf_total = shape_pmf * boundary_pdf.item() * pdf_factor
        if pdf_total <= 0:
            continue

        contrib = (color_diff * d_color).sum().item() / pdf_total

        # Accumulate gradient into shape parameters via Reynolds transport
        # (This is a simplified version - accumulates into a grad dict)
        key = ('shape', shape_id)
        if key not in grads:
            grads[key] = {'contrib': 0.0, 'normal': torch.zeros(2, device=device),
                         'boundary_pt': torch.zeros(2, device=device)}
        grads[key]['contrib'] += contrib

    return grads


def _gather_d_color(filter_type, filter_radius, d_color_image, weight_image,
                    width, height, pt):
    """Gather gradient color from d_color_image using filter kernel."""
    x = int(pt[0].item())
    y = int(pt[1].item())
    ri = int(math.ceil(filter_radius))
    d_color = torch.zeros(4, device=pt.device)
    for dy in range(-ri, ri + 1):
        for dx in range(-ri, ri + 1):
            xx = x + dx
            yy = y + dy
            if 0 <= xx < width and 0 <= yy < height:
                xc = xx + 0.5
                yc = yy + 0.5
                fw = compute_filter_weight(
                    filter_type, filter_radius,
                    torch.tensor(xc - pt[0].item(), device=pt.device),
                    torch.tensor(yc - pt[1].item(), device=pt.device))
                ws = weight_image[yy, xx]
                if ws > 0:
                    d_color += (fw / ws) * d_color_image[yy, xx]
    return d_color


def _eval_scene_color(pts_canvas, shapes, shape_groups, canvas_width, canvas_height,
                      background_image, device):
    """Evaluate scene color at canvas-space points. No grad."""
    N = pts_canvas.shape[0]
    if background_image is not None:
        bg = background_image.to(device)
        px = (pts_canvas[:, 0] / canvas_width * bg.shape[1]).long().clamp(0, bg.shape[1]-1)
        py = (pts_canvas[:, 1] / canvas_height * bg.shape[0]).long().clamp(0, bg.shape[0]-1)
        accum_color = bg[py, px, :3]
        accum_alpha = bg[py, px, 3]
    else:
        accum_color = torch.zeros(N, 3, device=device)
        accum_alpha = torch.zeros(N, device=device)

    for group_idx, sg in enumerate(shape_groups):
        shape_to_canvas = sg['shape_to_canvas']
        canvas_to_shape = torch.linalg.inv(shape_to_canvas)
        local_pts = xform_pt(canvas_to_shape, pts_canvas)
        use_even_odd = sg.get('use_even_odd_rule', True)

        fill_color_type = sg.get('fill_color_type')
        fill_color_data = sg.get('fill_color_data')
        stroke_color_type = sg.get('stroke_color_type')
        stroke_color_data = sg.get('stroke_color_data')

        # Check stroke first, then fill
        for is_stroke in [True, False]:
            if is_stroke:
                color_type = stroke_color_type
                color_data = stroke_color_data
            else:
                color_type = fill_color_type
                color_data = fill_color_data

            if color_type is None:
                continue

            hit = torch.zeros(N, device=device, dtype=torch.bool)
            for shape_id in sg['shape_ids']:
                sid = shape_id.item() if isinstance(shape_id, torch.Tensor) else shape_id
                st, sd = shapes[sid]
                sw = sd.get('stroke_width', torch.tensor(0.0))
                if isinstance(sw, (int, float)):
                    sw = torch.tensor(float(sw), device=device)

                if is_stroke and sw > 0:
                    hit = hit | within_distance_shape(st, sd, local_pts, sw)
                elif not is_stroke:
                    wn = compute_winding_number(st, sd, local_pts)
                    if use_even_odd:
                        hit = hit | (wn.abs() % 2 == 1)
                    else:
                        hit = hit | (wn != 0)

            if hit.any():
                rgba = sample_color(color_type, color_data, pts_canvas)
                na = torch.where(hit, rgba[:, 3], torch.zeros(N, device=device))
                nc = torch.where(hit.unsqueeze(-1), rgba[:, :3],
                                torch.zeros(N, 3, device=device))
                accum_color = accum_color * (1 - na.unsqueeze(-1)) + na.unsqueeze(-1) * nc
                accum_alpha = accum_alpha * (1 - na) + na

    safe_alpha = accum_alpha.clamp(min=1e-6)
    final_color = accum_color / safe_alpha.unsqueeze(-1)
    final_color = torch.where(accum_alpha.unsqueeze(-1) > 1e-6,
                              final_color, torch.zeros_like(final_color))
    return torch.cat([final_color, accum_alpha.unsqueeze(-1)], dim=-1)


# ============================================================================
# Conversion helpers: pydiffvg objects -> internal representation
# ============================================================================

def convert_shapes(pydiffvg_shapes, device):
    """Convert pydiffvg shape objects to internal representation.

    Returns: list of (shape_type, shape_data) tuples
    """
    import pydiffvg

    shapes = []
    for shape in pydiffvg_shapes:
        if isinstance(shape, pydiffvg.Circle):
            data = {
                'center': shape.center.to(device),
                'radius': shape.radius.to(device) if isinstance(shape.radius, torch.Tensor)
                         else torch.tensor(float(shape.radius), device=device),
                'stroke_width': shape.stroke_width.to(device) if isinstance(shape.stroke_width, torch.Tensor)
                               else torch.tensor(float(shape.stroke_width), device=device),
            }
            shapes.append((ShapeType.circle, data))
        elif isinstance(shape, pydiffvg.Ellipse):
            data = {
                'center': shape.center.to(device),
                'radius': shape.radius.to(device),
                'stroke_width': shape.stroke_width.to(device) if isinstance(shape.stroke_width, torch.Tensor)
                               else torch.tensor(float(shape.stroke_width), device=device),
            }
            shapes.append((ShapeType.ellipse, data))
        elif isinstance(shape, (pydiffvg.Path, pydiffvg.Polygon)):
            if isinstance(shape, pydiffvg.Polygon):
                if shape.is_closed:
                    ncp = torch.zeros(shape.points.shape[0], dtype=torch.int32)
                else:
                    ncp = torch.zeros(shape.points.shape[0] - 1, dtype=torch.int32)
            else:
                ncp = shape.num_control_points.to(torch.int32)
            data = {
                'points': shape.points.to(device),
                'num_control_points': ncp.to(device),
                'is_closed': shape.is_closed,
                'stroke_width': shape.stroke_width.to(device) if isinstance(shape.stroke_width, torch.Tensor)
                               else torch.tensor(float(shape.stroke_width), device=device),
                'use_distance_approx': getattr(shape, 'use_distance_approx', False),
            }
            # Handle per-point thickness
            if hasattr(shape, 'stroke_width') and isinstance(shape.stroke_width, torch.Tensor) and \
               len(shape.stroke_width.shape) > 0 and shape.stroke_width.shape[0] > 1:
                data['thickness'] = shape.stroke_width.to(device)
            shapes.append((ShapeType.path, data))
        elif isinstance(shape, pydiffvg.Rect):
            data = {
                'p_min': shape.p_min.to(device),
                'p_max': shape.p_max.to(device),
                'stroke_width': shape.stroke_width.to(device) if isinstance(shape.stroke_width, torch.Tensor)
                               else torch.tensor(float(shape.stroke_width), device=device),
            }
            shapes.append((ShapeType.rect, data))
    return shapes


def convert_shape_groups(pydiffvg_groups, device):
    """Convert pydiffvg ShapeGroup objects to internal representation."""
    import pydiffvg

    groups = []
    for sg in pydiffvg_groups:
        group = {
            'shape_ids': sg.shape_ids.to(device),
            'use_even_odd_rule': sg.use_even_odd_rule,
            'shape_to_canvas': sg.shape_to_canvas.to(device),
        }

        # Fill color
        if sg.fill_color is None:
            group['fill_color_type'] = None
            group['fill_color_data'] = None
        elif isinstance(sg.fill_color, torch.Tensor):
            group['fill_color_type'] = ColorType.constant
            group['fill_color_data'] = sg.fill_color.to(device)
        elif isinstance(sg.fill_color, pydiffvg.LinearGradient):
            group['fill_color_type'] = ColorType.linear_gradient
            group['fill_color_data'] = {
                'begin': sg.fill_color.begin.to(device),
                'end': sg.fill_color.end.to(device),
                'offsets': sg.fill_color.offsets.to(device),
                'stop_colors': sg.fill_color.stop_colors.to(device),
            }
        elif isinstance(sg.fill_color, pydiffvg.RadialGradient):
            group['fill_color_type'] = ColorType.radial_gradient
            group['fill_color_data'] = {
                'center': sg.fill_color.center.to(device),
                'radius': sg.fill_color.radius.to(device),
                'offsets': sg.fill_color.offsets.to(device),
                'stop_colors': sg.fill_color.stop_colors.to(device),
            }

        # Stroke color
        if sg.stroke_color is None:
            group['stroke_color_type'] = None
            group['stroke_color_data'] = None
        elif isinstance(sg.stroke_color, torch.Tensor):
            group['stroke_color_type'] = ColorType.constant
            group['stroke_color_data'] = sg.stroke_color.to(device)
        elif isinstance(sg.stroke_color, pydiffvg.LinearGradient):
            group['stroke_color_type'] = ColorType.linear_gradient
            group['stroke_color_data'] = {
                'begin': sg.stroke_color.begin.to(device),
                'end': sg.stroke_color.end.to(device),
                'offsets': sg.stroke_color.offsets.to(device),
                'stop_colors': sg.stroke_color.stop_colors.to(device),
            }
        elif isinstance(sg.stroke_color, pydiffvg.RadialGradient):
            group['stroke_color_type'] = ColorType.radial_gradient
            group['stroke_color_data'] = {
                'center': sg.stroke_color.center.to(device),
                'radius': sg.stroke_color.radius.to(device),
                'offsets': sg.stroke_color.offsets.to(device),
                'stop_colors': sg.stroke_color.stop_colors.to(device),
            }

        groups.append(group)
    return groups
