from dolfin import *


def distance_from_bdry_piece(facet_f, tag):
    """Solve Eikonal equation to get distance from tagged facets"""
    mesh = facet_f.mesh()
    V = FunctionSpace(mesh, 'CG', 1)
    bc = DirichletBC(V, Constant(0.0), facet_f, tag)

    u, v = TrialFunction(V), TestFunction(V)
    phi = Function(V)
    f = Constant(1.0)

    # Smooth initial guess
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx
    solve(a == L, phi, bc, solver_parameters={"linear_solver": "gmres"})

    # Eikonal equation with stabilization
    print("Solving stabilized Eikonal equation")
    eps = Constant(mesh.hmax() / 100)
    F = sqrt(inner(grad(phi), grad(phi))) * v * dx \
        - inner(f, v) * dx \
        + eps * inner(grad(phi), grad(v)) * dx

    solve(F == 0, phi, bc)

    return phi
