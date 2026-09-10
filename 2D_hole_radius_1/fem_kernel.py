"""
Shared FEniCSx 2-D plane-stress FEM kernel — hole plate under edge traction
===========================================================================
Single source of truth for the constitutive law and weak form used by both

  - fem_reference_solver.py   (standalone FEM -> CSV/VTU)
  - evaluate_gnn_vs_fem.py    (FEM for the GNN comparison)

The body is a square plate with circular holes cut out of it, meshed with linear
triangles. It is held by a ROLLER on x = x_min (ux = 0) plus a single PIN at the
top-left corner (uy = 0), and loaded on x = x_max by a uniform traction in +x.
Every hole rim is an exterior traction-free free surface.

Material models
---------------
  'LE'  — small-strain linear elasticity, PLANE STRESS via the reduced Lame
          parameter

              lambda_ps = 2 mu nu / (1 - nu)

          which bakes sigma_33 = 0 into the constitutive law:

              sigma = lambda_ps tr(eps) I + 2 mu eps,   eps = sym(grad u)

  'NH'  — compressible Neo-Hookean, finite strain. sigma_33 = 0 is imposed
          RIGOROUSLY by carrying the out-of-plane stretch F33 as an extra DG0
          unknown and minimising the FULL 3-D energy

              W = mu/2 (I1 - 3) - mu ln J + lambda/2 (ln J)^2

          over (u, F33), with I1 = tr(F2^T F2) + F33^2 and J = det(F2) F33. The
          F33 sub-field carries no external load, so dW/dF33 = 0 <=> sigma_33 = 0
          pointwise. This requires the FULL 3-D
          lambda = E nu / ((1+nu)(1-2nu)); the reduced lambda_ps would
          double-count the plane-stress correction.

The traction load
-----------------
The loaded edge carries the uniform dead traction

    t = (T, 0)

Two properties matter:

  * Its resultant is exactly F_x = T * (loaded edge length), computed from the
    actual mesh in ``applied_force`` so the reported number reflects the
    discretised edge rather than the analytic ideal.

  * It is a DEAD load: the traction vector is fixed in the reference frame and
    does not follow the material. It is therefore conservative — the external
    work is a genuine potential, which is what lets the PI-GNN minimise a total
    potential energy Pi = E_int - W_ext.

Stress output
    ``sigma_expr_fn(uh)`` returns the in-plane 2x2 Cauchy stress.

The kernel is deliberately agnostic about mesh loading, boundary-entity location
and output — callers locate the boundary entities and pass them in.
"""

import numpy as np
import ufl
import basix
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh, default_scalar_type
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver


def lame_parameters(E_func, nu_func, material_model):
    """
    Per-element Lame fields (mu, lam) as UFL expressions of the DG0 E/nu fields.

    LE -> reduced plane-stress lambda_ps; NH -> full 3-D lambda (see the module
    docstring). mu = E/(2(1+nu)) is identical for both.
    """
    mu = E_func / (2.0 * (1.0 + nu_func))
    if str(material_model).upper() == 'NH':
        lam = E_func * nu_func / ((1.0 + nu_func) * (1.0 - 2.0 * nu_func))   # full 3-D
    else:
        lam = 2.0 * mu * nu_func / (1.0 - nu_func)                          # plane stress
    return mu, lam


# Backwards-compatible alias (the name the previous kernel exported).
plane_stress_lame = lame_parameters


def build_traction_measure(domain, fdim, loaded_facets):
    """Build a surface measure with the loaded edge tagged as subdomain 2.

    Args:
        domain: DOLFINx mesh of the plate.
        fdim: Facet dimension (``tdim - 1``, i.e. 1 for a triangular mesh).
        loaded_facets: Integer array of facet indices forming the loaded edge,
            shape (n_facets,).

    Returns:
        ufl.Measure: A ``ds`` measure carrying facet tags, where ``ds(2)``
        integrates over the loaded edge only.
    """
    f_idx  = np.asarray(loaded_facets, dtype=np.int32)
    f_mark = np.full_like(f_idx, 2)
    sort_  = np.argsort(f_idx)
    facet_tag = dmesh.meshtags(domain, fdim, f_idx[sort_], f_mark[sort_])
    return ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)


def edge_traction(domain, traction):
    """The uniform dead-load traction t = (T, 0) as a UFL vector constant."""
    return fem.Constant(domain, default_scalar_type((float(traction), 0.0)))


def loaded_edge_length(domain, ds_measure, tag=2):
    """Length of the discretised loaded edge, ``Int_tag 1 ds``."""
    form = fem.form(fem.Constant(domain, default_scalar_type(1.0)) * ds_measure(tag))
    return float(domain.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM))


def applied_force(domain, traction, ds_measure, tag=2):
    """
    Resultant force the loaded edge actually applies, ``Int_tag T ds``.

    Assembled over the discretised edge rather than taken as T * L_analytic, so
    the number reported is the load the solver really sees. For a straight edge
    the two agree to machine precision; the assembled form is what stays correct
    if the loaded boundary is ever curved or partially masked.
    """
    T = fem.Constant(domain, default_scalar_type(float(traction)))
    form = fem.form(T * ds_measure(tag))
    return float(domain.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM))


def solve_composite(domain, E_func, nu_func, material_model,
                    *, left_facets, corner_verts, right_facets,
                    traction, fdim, n_load_steps=5, verbose=True):
    """
    Solve plane-stress LE or rigorous finite-strain plane-stress NH on `domain`.

    Boundary conditions
        ux = 0  on `left_facets`   (roller; facet entities)
        uy = 0  at `corner_verts`  (pin; dim-0 vertex entities — a corner is a
                                    vertex, not a facet, and a facet search for
                                    it finds nothing, leaving uy with an
                                    unconstrained rigid-body mode)
        t = (traction, 0) on `right_facets`  (Neumann)
        everything else, hole rims included: free (natural BC, traction-free)

    Parameters
    ----------
    E_func, nu_func : DG0 fem.Function   per-element Young's modulus / Poisson ratio
    material_model  : 'LE' | 'NH'
    traction        : float              uniform traction magnitude on the right edge
    fdim            : int                facet dimension (= tdim - 1 = 1)
    n_load_steps    : int                NH incremental-loading steps (ignored for LE)

    Returns
    -------
    (uh, sigma_expr_fn)
        uh            : displacement Function (collapsed P1 vector for NH).
        sigma_expr_fn : callable uh -> 2x2 in-plane Cauchy stress UFL expression
                        (for NH it closes over the converged F33 field).
    """
    model = str(material_model).upper()
    mu_ufl, lam_ufl = lame_parameters(E_func, nu_func, model)
    gdim = domain.geometry.dim
    if gdim != 2:
        raise ValueError(f"This kernel solves the 2-D plate; got gdim={gdim}.")

    ds   = build_traction_measure(domain, fdim, right_facets)
    T_r  = edge_traction(domain, traction)
    domain.topology.create_connectivity(0, domain.topology.dim)

    if verbose:
        L_e = loaded_edge_length(domain, ds)
        F_x = applied_force(domain, traction, ds)
        print(f"    Loaded edge length : {L_e:.6e}")
        print(f"    Applied force Fx   : {F_x:.6e}  (= T x L)")

    # ── Linear elasticity (plane stress) ─────────────────────────────────────
    if model != 'NH':
        V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

        def small_strain(u):
            """Infinitesimal strain tensor eps = sym(grad u) (2x2 UFL expr)."""
            return ufl.sym(ufl.grad(u))

        def linear_elastic_stress(u):
            """Cauchy stress sigma = lambda_ps tr(eps) I + 2 mu eps."""
            return (lam_ufl * ufl.nabla_div(u) * ufl.Identity(gdim)
                    + 2.0 * mu_ufl * small_strain(u))

        left_dofs_x = fem.locate_dofs_topological(V.sub(0), fdim, left_facets)
        bc_ux       = fem.dirichletbc(default_scalar_type(0), left_dofs_x, V.sub(0))
        corner_dy   = fem.locate_dofs_topological(V.sub(1), 0, corner_verts)
        bc_uy       = fem.dirichletbc(default_scalar_type(0), corner_dy, V.sub(1))

        u_t, v_t = ufl.TrialFunction(V), ufl.TestFunction(V)
        a = ufl.inner(linear_elastic_stress(u_t), small_strain(v_t)) * ufl.dx
        L = ufl.dot(T_r, v_t) * ds(2)
        problem = LinearProblem(
            a, L, bcs=[bc_ux, bc_uy],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                           "pc_factor_mat_solver_type": "mumps"},
        )
        uh = problem.solve()
        return uh, (lambda uh_: linear_elastic_stress(uh_))

    # ── Neo-Hookean (rigorous finite-strain plane stress) ────────────────────
    from petsc4py import PETSc

    cell = domain.basix_cell()
    el_u = basix.ufl.element("Lagrange", cell, 1, shape=(gdim,))
    el_s = basix.ufl.element("DG", cell, 0)
    W    = fem.functionspace(domain, basix.ufl.mixed_element([el_u, el_s]))

    w = fem.Function(W)
    w.sub(1).interpolate(lambda x: np.ones(x.shape[1]))   # F33 = 1 (undeformed)
    w.x.scatter_forward()
    u_m, s33 = ufl.split(w)

    I2    = ufl.Identity(2)
    F2    = I2 + ufl.grad(u_m)
    J2    = ufl.det(F2)
    I1_2d = ufl.tr(F2.T * F2)
    J3    = J2 * s33                       # full 3-D Jacobian
    I1_3d = I1_2d + s33**2                 # full 3-D first invariant
    lnJ   = ufl.ln(J3)
    psi   = (mu_ufl / 2.0) * (I1_3d - 3.0) - mu_ufl * lnJ + (lam_ufl / 2.0) * lnJ**2
    Pi    = psi * ufl.dx

    v      = ufl.TestFunction(W)
    v_u, _ = ufl.split(v)                  # traction couples to u only
    F_res  = ufl.derivative(Pi, w, v) - ufl.dot(T_r, v_u) * ds(2)
    J_tang = ufl.derivative(F_res, w)

    Wux, Wuy = W.sub(0).sub(0), W.sub(0).sub(1)
    bc_ux = fem.dirichletbc(default_scalar_type(0),
                            fem.locate_dofs_topological(Wux, fdim, left_facets), Wux)
    bc_uy = fem.dirichletbc(default_scalar_type(0),
                            fem.locate_dofs_topological(Wuy, 0, corner_verts), Wuy)

    problem = NonlinearProblem(F_res, w, bcs=[bc_ux, bc_uy], J=J_tang)
    solver  = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol   = 1e-8    # relative Newton residual tolerance (dimensionless)
    solver.atol   = 1e-10   # absolute Newton residual tolerance (force units)
    solver.max_it = 50      # max Newton iterations per load increment
    # Handle a failed step here (by cutting the load increment) instead of
    # letting PETSc raise: an over-large first increment is a recoverable
    # stepping problem, not a broken model.
    solver.error_on_nonconvergence = False

    # Robust direct linear solve (mixed DG0 block -> LU/MUMPS)
    ksp  = solver.krylov_solver
    opts = PETSc.Options()
    pre  = ksp.getOptionsPrefix() or ""
    opts[f"{pre}ksp_type"] = "preonly"
    opts[f"{pre}pc_type"]  = "lu"
    opts[f"{pre}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    # ── Adaptive incremental loading ─────────────────────────────────────────
    # Walk the traction 0 -> T. If a step fails to converge, halve the increment
    # and retry from the last converged state (w is left holding it, since a
    # non-converged Newton solve does not restore the previous iterate — so the
    # state is explicitly cached and rolled back).
    T_full = float(traction)
    frac_now = 0.0
    d_frac = 1.0 / max(1, int(n_load_steps))
    w_last = w.x.array.copy()
    n_cuts = 0
    MAX_CUTS = 8          # max successive halvings before giving up

    while frac_now < 1.0 - 1e-12:
        frac_try = min(1.0, frac_now + d_frac)
        T_r.value[:] = (frac_try * T_full, 0.0)
        n_iter, converged = solver.solve(w)

        if converged:
            frac_now = frac_try
            w_last = w.x.array.copy()
            if verbose:
                print(f"    NH load step T={frac_now * T_full:.6g}: {n_iter} iter")
            # Grow the step back gently after a cut, so one hard patch of the
            # loading path does not force tiny steps all the way to full load.
            if n_cuts > 0 and n_iter <= 4:
                d_frac = min(2.0 * d_frac, 1.0 / max(1, int(n_load_steps)))
        else:
            w.x.array[:] = w_last          # roll back to the last converged state
            w.x.scatter_forward()
            d_frac *= 0.5
            n_cuts += 1
            if verbose:
                print(f"    NH step did not converge at T={frac_try * T_full:.6g}; "
                      f"cutting increment to {d_frac:.5f} (cut {n_cuts}/{MAX_CUTS})")
            if n_cuts > MAX_CUTS:
                raise RuntimeError(
                    f"Neo-Hookean solve failed to converge past "
                    f"T={frac_now * T_full:.6g} after {MAX_CUTS} increment cuts. "
                    f"The applied traction is likely beyond what this "
                    f"mesh/material can carry — reduce TRACTION_MAGNITUDE or "
                    f"refine the mesh.")

    uh      = w.sub(0).collapse()
    s33_sol = w.sub(1).collapse()
    if verbose:
        print(f"    NH F33 (out-of-plane stretch): "
              f"[{s33_sol.x.array.min():.5f}, {s33_sol.x.array.max():.5f}]")

    def neo_hookean_stress(uh_):
        """Cauchy stress of the compressible Neo-Hookean model, in-plane block.

        sigma = (1/J) ( mu (B2 - I) + lambda ln J I ),   B2 = F2 F2^T

        with the FULL 3-D J and lambda; sigma_33 = 0 holds by construction
        because F33 came from the converged mixed solve.
        """
        F_c   = I2 + ufl.grad(uh_)
        J3_c  = ufl.det(F_c) * s33_sol
        lnJ_c = ufl.ln(J3_c)
        B2    = F_c * F_c.T
        return (1.0 / J3_c) * (mu_ufl * (B2 - I2) + lam_ufl * lnJ_c * I2)

    return uh, neo_hookean_stress
