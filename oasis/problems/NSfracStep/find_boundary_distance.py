from dolfin import *


def find_distance(mesh, ignore_domain):
    """
    mesh: Mesh to calculate distance on
    Ignore_domain: Part of boundary to be ignored, specified with e.g. AutoSubDomain()
    """
    if MPI.comm_world.Get_size() == 1:
        bmesh = BoundaryMesh(mesh, 'exterior')
        cell_f = MeshFunction('size_t', bmesh, 0)
        ignore_domain.mark(cell_f, 1)

        bmesh_sub = SubMesh(bmesh, cell_f, 0)
        File('submesh1.xml') << bmesh_sub

    # FIXME: Does not work in parallel
    if MPI.comm_world.Get_size() >= 1:
        if MPI.comm_world.Get_size() > 1:
            bmesh_sub = Mesh("submesh1.xml")

        tree = bmesh_sub.bounding_box_tree()

        V = FunctionSpace(mesh, 'CG', 1)
        v_2_d = V.dofmap().entity_dofs(mesh, 0)

        bdry_distance = Function(V)
        values = bdry_distance.vector().get_local(v_2_d)
        for index, vertex in enumerate(vertices(mesh)):
            w, d = tree.compute_closest_entity(vertex.point())
            values[v_2_d[index]] = d
        bdry_distance.vector().set_local(values)
        bdry_distance.vector().apply('insert')

    return bdry_distance.vector()
