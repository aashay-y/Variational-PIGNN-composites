"""
Shared FEniCSx 3-D composite FEM kernel — clamped/twisted rod
=============================================================
Single source of truth for the constitutive law and weak form used by both

  - fem_reference_solver.py                  (standalone FEM -> CSV/VTU)
  - evaluate_gnn_vs_fem.py (FEM for the GNN comparison)

The body is a circular rod (radius R, length H) with a straight prismatic
inclusion, meshed with linear tetrahedra.  It is CLAMPED at z = 0 and loaded on
z = H by a tangential surface traction that applies a pure torque about the rod
axis.  There is no plane-stress or plane-strain assumption anywhere: this is the
full three-dimensional problem, so all six stress components are live and
sigma_zz, sigma_xz, sigma_yz are genuine unknowns rather than recovered ones.

Material models
---------------
  'NH'  — compressible Neo-Hookean, finite strain (the default; torsion to 50 %
          shear is far outside the linear range):

              W = mu/2 (I1 - 3) - mu ln J + lambda/2 (ln J)^2
              I1 = tr(F^T F),   J = det F,   F = I + grad u

          Fully 3-D — no out-of-plane condensation, unlike the plane-strain
          formulation this kernel used previously.

  'LE'   — small-strain linear elasticity, sigma = lambda tr(eps) I + 2 mu eps
          with eps = sym(grad u).  Kept as a reference/debug option; at the twist
          magnitudes this problem is posed at it is NOT physically valid, and
          the summary says so.

Both use the standard 3-D Lame parameters
    mu     = E / (2(1+nu))
    lambda = E nu / ((1+nu)(1-2nu))

The torsion load
----------------
The loaded face carries the dead traction

    t(X) = (tau / R) * ( -(y - cy),  (x - cx),  0 )

i.e. tangential, growing linearly with radius, vanishing on the axis.  Two
properties matter:

  * Its resultant force is zero by symmetry, so the load is a PURE COUPLE.  The
    torque it applies about the axis is exactly

        M_z = (tau / R) * Int_A r^2 dA  =  tau * pi * R^3 / 2   (for a disk)

    which is computed from the actual mesh in ``applied_torque`` so the reported
    number reflects the discretised face, not the analytic ideal.

  * It is a DEAD load: the traction vector is fixed in the reference frame and
    does not rotate with the material.  It is therefore conservative (the
    external work is a genuine potential, which is what lets the PI-GNN minimise
    a total potential energy), but at large twist it is not the same thing as a
    follower torque.  The twist angle is an OUTPUT of the applied torque, not an
    input.

Stress output
    ``sigma_expr_fn(uh)`` returns the FULL 3x3 Cauchy stress.

The kernel is deliberately agnostic about mesh loading, boundary-entity location
and output — callers locate the boundary entities and pass them in.
"""

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh, default_scalar_type
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver


def lame_parameters(E_func, nu_func, material_model=None):
    """
    Per-element Lame fields (mu, lam) as UFL expressions of the DG0 E/nu fields.

    Standard 3-D parameters; both material models use them unmodified, because
    the problem is fully three-dimensional and no dimensional reduction applies.
    ``material_model`` is accepted only for call-site symmetry.
    """
    mu  = E_func / (2.0 * (1.0 + nu_func))
    lam = E_func * nu_func / ((1.0 + nu_func) * (1.0 - 2.0 * nu_func))
    return mu, lam


def build_traction_measure(domain, fdim, loaded_facets):
    """Build a surface measure with the loaded end face tagged as subdomain 2.

    Args:
        domain: DOLFINx mesh of the rod.
        fdim: Facet dimension (``tdim - 1``, i.e. 2 for a tetrahedral mesh).
        loaded_facets: Integer array of facet indices forming the loaded end
            face, shape (n_facets,).

    Returns:
        ufl.Measure: A ``ds`` measure carrying facet tags, where ``ds(2)``
        integrates over the loaded face only.
    """
    f_idx  = np.asarray(loaded_facets, dtype=np.int32)
    f_mark = np.full_like(f_idx, 2)
    sort_  = np.argsort(f_idx)
    facet_tag = dmesh.meshtags(domain, fdim, f_idx[sort_], f_mark[sort_])
    return ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)


def torsion_traction(domain, traction, axis, plate_radius):
    """
    The tangential dead-load traction field t(X) as a UFL vector expression.

    t = (tau / R) * (-(y - cy), (x - cx), 0)

    Linear in position and regular on the axis (it vanishes there), so no
    singularity has to be special-cased — unlike a t ~ 1/r or a normalised
    e_theta field, both of which blow up or are undefined at r = 0.
    """
    x  = ufl.SpatialCoordinate(domain)
    cx, cy = float(axis[0]), float(axis[1])
    scale = float(traction) / float(plate_radius)
    return scale * ufl.as_vector([-(x[1] - cy), (x[0] - cx), 0.0])


def applied_torque(domain, traction, axis, plate_radius, ds_measure, tag=2):
    """
    Torque about the rod axis actually applied by the traction on the tagged
    face, integrated over the DISCRETISED face:

        M_z = Int ( (x-cx) t_y - (y-cy) t_x ) dA
            = (tau/R) Int r^2 dA

    Reported alongside the analytic tau*pi*R^3/2 so the polygonal-rim deficit of
    the mesh is visible rather than hidden.
    """
    x = ufl.SpatialCoordinate(domain)
    cx, cy = float(axis[0]), float(axis[1])
    t = torsion_traction(domain, traction, axis, plate_radius)
    m = (x[0] - cx) * t[1] - (x[1] - cy) * t[0]
    form = fem.form(m * ds_measure(tag))
    return float(domain.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM))


def loaded_face_area(domain, ds_measure, tag=2):
    """Area of the tagged (loaded) face, from the mesh."""
    form = fem.form(fem.Constant(domain, default_scalar_type(1.0)) * ds_measure(tag))
    return float(domain.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM))


def solve_composite(domain, E_func, nu_func, material_model,
                    *, fixed_facets, loaded_facets, traction,
                    axis, plate_radius, fdim,
                    n_load_steps=10, verbose=True):
    """
    Solve 3-D LE or finite-strain NH torsion of the composite rod on `domain`.

    Boundary conditions
        u = 0                          on `fixed_facets`   (clamped end, z = 0)
        t = (tau/R)(-(y-cy), (x-cx), 0) on `loaded_facets`  (twisted end, z = H)
        lateral cylindrical surface: free (natural BC, traction-free)

    The clamped face removes all six rigid-body modes, so the system is
    non-singular with no extra constraint.

    Parameters
    ----------
    E_func, nu_func : DG0 fem.Function   per-element Young's modulus / Poisson ratio
    material_model  : 'NH' | 'LE'
    traction        : float              tangential traction magnitude tau at r = R
    axis            : (float, float)     rod axis (cx, cy), same units as the mesh
    plate_radius    : float              rod radius R, same units as the mesh
    fdim            : int                facet dimension (= tdim - 1 = 2)
    n_load_steps    : int                NH incremental-loading steps (ignored for LE)

    Returns
    -------
    (uh, sigma_expr_fn)
        uh            : displacement Function (3 components).
        sigma_expr_fn : callable uh -> full 3x3 Cauchy stress UFL expression.
    """
    model = str(material_model).upper()
    mu_ufl, lam_ufl = lame_parameters(E_func, nu_func, model)
    gdim = domain.geometry.dim
    if gdim != 3:
        raise ValueError(f"This kernel solves the 3-D rod; got gdim={gdim}.")

    ds = build_traction_measure(domain, fdim, loaded_facets)
    domain.topology.create_connectivity(fdim, domain.topology.dim)

    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

    # Clamped end: all three components zero.
    u_clamp = fem.Function(V)
    u_clamp.x.array[:] = 0.0
    bc_clamp = fem.dirichletbc(
        u_clamp, fem.locate_dofs_topological(V, fdim, fixed_facets))

    # Load-factor multiplier: 1.0 for LE, ramped 0 -> 1 for the NH Newton solve.
    load_factor = fem.Constant(domain, default_scalar_type(1.0))
    t_ref = torsion_traction(domain, traction, axis, plate_radius)
    t_vec = load_factor * t_ref

    if verbose:
        M_z = applied_torque(domain, traction, axis, plate_radius, ds)
        A_l = loaded_face_area(domain, ds)
        M_analytic = traction * np.pi * plate_radius ** 3 / 2.0
        print(f"    Loaded face area : {A_l:.6e}  (analytic pi*R^2 = "
              f"{np.pi * plate_radius**2:.6e})")
        print(f"    Applied torque   : {M_z:.6e}  (analytic tau*pi*R^3/2 = "
              f"{M_analytic:.6e})")

    # ── Linear elasticity (small strain, full 3-D) ───────────────────────────
    if model != 'NH':
        def small_strain(u):
            """Infinitesimal strain tensor eps = sym(grad u) (3x3 UFL expr)."""
            return ufl.sym(ufl.grad(u))

        def linear_elastic_stress(u):
            """Cauchy stress sigma = lambda tr(eps) I + 2 mu eps (3x3 UFL expr)."""
            return (lam_ufl * ufl.div(u) * ufl.Identity(gdim)
                    + 2.0 * mu_ufl * small_strain(u))

        u_t, v_t = ufl.TrialFunction(V), ufl.TestFunction(V)
        a = ufl.inner(linear_elastic_stress(u_t), small_strain(v_t)) * ufl.dx
        L = ufl.dot(t_vec, v_t) * ds(2)
        problem = LinearProblem(
            a, L, bcs=[bc_clamp],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                           "pc_factor_mat_solver_type": "mumps"},
        )
        uh = problem.solve()
        return uh, (lambda uh_: linear_elastic_stress(uh_))

    # ── Neo-Hookean (finite strain, full 3-D) ────────────────────────────────
    from petsc4py import PETSc

    u = fem.Function(V)

    I3   = ufl.Identity(3)
    F_d  = I3 + ufl.grad(u)
    J    = ufl.det(F_d)
    I1   = ufl.tr(F_d.T * F_d)
    lnJ  = ufl.ln(J)
    psi  = (mu_ufl / 2.0) * (I1 - 3.0) - mu_ufl * lnJ + (lam_ufl / 2.0) * lnJ**2
    Pi   = psi * ufl.dx

    v      = ufl.TestFunction(V)
    F_res  = ufl.derivative(Pi, u, v) - ufl.dot(t_vec, v) * ds(2)
    J_tang = ufl.derivative(F_res, u)

    problem = NonlinearProblem(F_res, u, bcs=[bc_clamp], J=J_tang)
    solver  = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol   = 1e-8    # relative Newton residual tolerance (dimensionless)
    solver.atol   = 1e-10   # absolute Newton residual tolerance (force units)
    solver.max_it = 50      # max Newton iterations per load increment
    # Handle a failed step here (by cutting the load increment) instead of
    # letting PETSc raise: at 50 % shear an over-large first increment is a
    # recoverable stepping problem, not a broken model.
    solver.error_on_nonconvergence = False

    # Robust direct linear solve
    ksp  = solver.krylov_solver
    opts = PETSc.Options()
    pre  = ksp.getOptionsPrefix() or ""
    opts[f"{pre}ksp_type"] = "preonly"
    opts[f"{pre}pc_type"]  = "lu"
    opts[f"{pre}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    # ── Adaptive incremental loading ─────────────────────────────────────────
    # Walk the load factor 0 -> 1. If a step fails to converge, halve the
    # increment and retry from the last converged state (u is left holding it,
    # since a non-converged Newton solve does not restore the previous iterate —
    # so the state is explicitly cached and rolled back).
    lam_now = 0.0
    d_lam = 1.0 / max(1, int(n_load_steps))
    u_last = u.x.array.copy()
    n_cuts = 0
    MAX_CUTS = 8          # max successive halvings of the load increment before giving up

    while lam_now < 1.0 - 1e-12:
        lam_try = min(1.0, lam_now + d_lam)
        load_factor.value = lam_try
        n_iter, converged = solver.solve(u)

        if converged:
            lam_now = lam_try
            u_last = u.x.array.copy()
            if verbose:
                print(f"    NH load step lambda={lam_now:.4f} "
                      f"(tau={lam_now * traction:.4g}): {n_iter} iter")
            # Grow the step back gently after a cut, so one hard patch of the
            # loading path does not force tiny steps all the way to full load.
            if n_cuts > 0 and n_iter <= 4:
                d_lam = min(2.0 * d_lam, 1.0 / max(1, int(n_load_steps)))
        else:
            u.x.array[:] = u_last          # roll back to the last converged state
            u.x.scatter_forward()
            d_lam *= 0.5
            n_cuts += 1
            if verbose:
                print(f"    NH step did not converge at lambda={lam_try:.4f}; "
                      f"cutting increment to {d_lam:.5f} (cut {n_cuts}/{MAX_CUTS})")
            if n_cuts > MAX_CUTS:
                raise RuntimeError(
                    f"Neo-Hookean solve failed to converge past lambda={lam_now:.4f} "
                    f"after {MAX_CUTS} increment cuts. The applied torque is likely "
                    f"beyond what this mesh/material can carry — reduce "
                    f"TRACTION_MAGNITUDE or refine the mesh.")

    def neo_hookean_stress(uh_):
        """Cauchy stress of the compressible Neo-Hookean model.

        Args:
            uh_: Displacement Function (3 components) to evaluate the stress at.

        Returns:
            UFL expression for the full 3x3 Cauchy stress tensor.
        """
        # sigma = (1/J) ( mu (B - I) + lambda ln J I ),   B = F F^T
        F_c   = I3 + ufl.grad(uh_)
        J_c   = ufl.det(F_c)
        lnJ_c = ufl.ln(J_c)
        B     = F_c * F_c.T
        return (1.0 / J_c) * (mu_ufl * (B - I3) + lam_ufl * lnJ_c * I3)

    return u, neo_hookean_stress
