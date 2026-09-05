import time
import math
import heapq
import logging
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

class Mesh:
    """Minimal mesh container (replaces psbody.mesh.Mesh)."""
    def __init__(self, v, f):
        self.v = np.array(v, dtype=np.float64)
        self.f = np.array(f, dtype=np.uint32)

def row(A):
    return A.reshape((1, -1))

def col(A):
    return A.reshape((-1, 1))

def get_vert_connectivity(mesh_v, mesh_f):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12."""

    vpv = sp.csc_matrix((len(mesh_v),len(mesh_v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh_f[:,i]
        JS = mesh_f[:,(i+1)%3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv

def get_vertices_per_edge(mesh_v, mesh_f):
    """Returns an Ex2 array of adjacencies between vertices, where
    each element in the array is a vertex index. Each edge is included
    only once. If output of get_faces_per_edge is provided, this is used to
    avoid call to get_vert_connectivity()"""

    vc = sp.coo_matrix(get_vert_connectivity(mesh_v, mesh_f))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:,0] < result[:,1]] # for uniqueness

    return result


def vertex_quadrics(mesh):
    """Computes a quadric for each vertex in the Mesh.

    Returns:
       v_quadrics: an (N x 4 x 4) array, where N is # vertices.
    """

    # Allocate quadrics
    v_quadrics = np.zeros((len(mesh.v), 4, 4,))

    # For each face...
    for f_idx in range(len(mesh.f)):

        # Compute normalized plane equation for that face
        vert_idxs = mesh.f[f_idx]
        verts = np.hstack((mesh.v[vert_idxs], np.array([1, 1, 1]).reshape(-1, 1)))
        u, s, v = np.linalg.svd(verts)
        eq = v[-1, :].reshape(-1, 1)
        eq = eq / (np.linalg.norm(eq[0:3]))

        # Add the outer product of the plane equation to the
        # quadrics of the vertices for this face
        for k in range(3):
            v_quadrics[mesh.f[f_idx, k], :, :] += np.outer(eq, eq)

    return v_quadrics

def _get_sparse_transform(faces, num_original_verts):
    verts_left = np.unique(faces.flatten())
    IS = np.arange(len(verts_left))
    JS = verts_left
    data = np.ones(len(JS))

    mp = np.arange(0, np.max(faces.flatten()) + 1)
    mp[JS] = IS
    new_faces = mp[faces.copy().flatten()].reshape((-1, 3))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij), shape=(len(verts_left) , num_original_verts ))

    return (new_faces, mtx)

def qslim_decimator_transformer(mesh, factor=None, n_verts_desired=None):
    """Return a simplified version of this mesh.

    A Qslim-style approach is used here.

    :param factor: fraction of the original vertices to retain
    :param n_verts_desired: number of the original vertices to retain
    :returns: new_faces: An Fx3 array of faces, mtx: Transformation matrix
    """

    if factor is None and n_verts_desired is None:
        raise Exception('Need either factor or n_verts_desired.')

    if n_verts_desired is None:
        n_verts_desired = math.ceil(len(mesh.v) * factor)

    decimation_start = time.perf_counter()
    logger.info(
        "QSlim decimation started: vertices=%d, faces=%d, target_vertices=%d",
        len(mesh.v),
        len(mesh.f),
        n_verts_desired,
    )

    Qv = vertex_quadrics(mesh)

    # fill out a sparse matrix indicating vertex-vertex adjacency
    # from psbody.mesh.topology.connectivity import get_vertices_per_edge
    vert_adj = get_vertices_per_edge(mesh.v, mesh.f)
    # vert_adj = sp.lil_matrix((len(mesh.v), len(mesh.v)))
    # for f_idx in range(len(mesh.f)):
    #     vert_adj[mesh.f[f_idx], mesh.f[f_idx]] = 1

    vert_adj = sp.csc_matrix((vert_adj[:, 0] * 0 + 1, (vert_adj[:, 0], vert_adj[:, 1])), shape=(len(mesh.v), len(mesh.v)))
    vert_adj = vert_adj + vert_adj.T
    vert_adj = vert_adj.tocoo()

    def collapse_cost(Qv, r, c, v):
        Qsum = Qv[r, :, :] + Qv[c, :, :]
        p1 = np.vstack((v[r].reshape(-1, 1), np.array([1]).reshape(-1, 1)))
        p2 = np.vstack((v[c].reshape(-1, 1), np.array([1]).reshape(-1, 1)))

        destroy_c_cost = p1.T.dot(Qsum).dot(p1)
        destroy_r_cost = p2.T.dot(Qsum).dot(p2)
        result = {
            'destroy_c_cost': destroy_c_cost,
            'destroy_r_cost': destroy_r_cost,
            'collapse_cost': min([destroy_c_cost, destroy_r_cost]),
            'Qsum': Qsum}
        return result

    # construct a queue of edges with costs
    queue = []
    for k in range(vert_adj.nnz):
        r = vert_adj.row[k]
        c = vert_adj.col[k]

        if r > c:
            continue

        cost = collapse_cost(Qv, r, c, mesh.v)['collapse_cost']
        heapq.heappush(queue, (cost, (r, c)))

    last_log = time.perf_counter()
    # decimate
    collapse_list = []
    nverts_total = len(mesh.v)
    faces = mesh.f.copy()
    nverts_total_ = nverts_total
    while nverts_total > n_verts_desired:

        #if nverts_total % 100 == 0 and nverts_total_ != nverts_total:            
        #    nverts_total_ = nverts_total
        #    print(f"{time.time() - start}: {nverts_total}")
        #    start = time.time()

        e = heapq.heappop(queue)
        r = e[1][0]
        c = e[1][1]
        if r == c:
            continue

        cost = collapse_cost(Qv, r, c, mesh.v)
        if cost['collapse_cost'] > e[0]:
            heapq.heappush(queue, (cost['collapse_cost'], e[1]))
            # print 'found outdated cost, %.2f < %.2f' % (e[0], cost['collapse_cost'])
            continue
        else:

            # update old vert idxs to new one,
            # in queue and in face list
            if cost['destroy_c_cost'] < cost['destroy_r_cost']:
                to_destroy = c
                to_keep = r
            else:
                to_destroy = r
                to_keep = c

            collapse_list.append([to_keep, to_destroy])

            # in our face array, replace "to_destroy" vertidx with "to_keep" vertidx
            np.place(faces, faces == to_destroy, to_keep)

            # same for queue
            which1 = [idx for idx in range(len(queue)) if queue[idx][1][0] == to_destroy]
            which2 = [idx for idx in range(len(queue)) if queue[idx][1][1] == to_destroy]
            for k in which1:
                queue[k] = (queue[k][0], (to_keep, queue[k][1][1]))
            for k in which2:
                queue[k] = (queue[k][0], (queue[k][1][0], to_keep))

            Qv[r, :, :] = cost['Qsum']
            Qv[c, :, :] = cost['Qsum']

            a = faces[:, 0] == faces[:, 1]
            b = faces[:, 1] == faces[:, 2]
            c = faces[:, 2] == faces[:, 0]

            # remove degenerate faces
            def logical_or3(x, y, z):
                return np.logical_or(x, np.logical_or(y, z))

            faces_to_keep = np.logical_not(logical_or3(a, b, c))
            faces = faces[faces_to_keep, :].copy()

        nverts_total = (len(np.unique(faces.flatten())))
        now = time.perf_counter()
        if now - last_log > 15:
            logger.info(
                "QSlim decimation progress: vertices=%d/%d, target=%d, faces=%d, queue=%d, elapsed=%.1fs",
                nverts_total,
                len(mesh.v),
                n_verts_desired,
                len(faces),
                len(queue),
                now - decimation_start,
            )
            last_log = now

    new_faces, mtx = _get_sparse_transform(faces, len(mesh.v))
    logger.info(
        "QSlim decimation finished: vertices=%d -> %d, faces=%d, elapsed=%.2fs",
        len(mesh.v),
        len(np.unique(new_faces.flatten())),
        len(new_faces),
        time.perf_counter() - decimation_start,
    )
    return new_faces, mtx


def setup_deformation_transfer(source, target, use_normals=False):
    import trimesh

    start = time.perf_counter()
    logger.info(
        "Computing deformation transfer: source_vertices=%d, source_faces=%d, target_vertices=%d",
        len(source.v),
        len(source.f),
        len(target.v),
    )
    source_trimesh = trimesh.Trimesh(vertices=source.v, faces=source.f, process=False)
    closest_points, _, nearest_faces = source_trimesh.nearest.on_surface(target.v)
    nearest_faces = nearest_faces.astype(np.int64)

    rows     = np.zeros(3 * target.v.shape[0])
    cols     = np.zeros(3 * target.v.shape[0])
    coeffs_v = np.zeros(3 * target.v.shape[0])

    for i in range(target.v.shape[0]):
        f_id      = nearest_faces[i]
        nearest_f = source.f[f_id]
        p         = closest_points[i]

        rows[3 * i:3 * i + 3] = i
        cols[3 * i:3 * i + 3] = nearest_f

        # Barycentric coordinates of p w.r.t. the triangle
        A = source.v[nearest_f].T          # (3, 3): columns are triangle vertices
        bary, *_ = np.linalg.lstsq(A, p, rcond=None)
        coeffs_v[3 * i:3 * i + 3] = bary

    matrix = sp.csc_matrix((coeffs_v, (rows, cols)), shape=(target.v.shape[0], source.v.shape[0]))
    logger.info("Deformation transfer computed in %.2fs", time.perf_counter() - start)
    return matrix


def generate_transform_matrices(mesh, factors):
    """Generates len(factors) meshes, each of them is scaled by factors[i] and
       computes the transformations between them.

    Returns:
       M: a set of meshes downsampled from mesh by a factor specified in factors.
       A: Adjacency matrix for each of the meshes
       D: Downsampling transforms between each of the meshes
       U: Upsampling transforms between each of the meshes
    """

    factors = list(map(lambda x: 1.0 / x, factors))
    logger.info(
        "Generating transform matrices: initial_vertices=%d, initial_faces=%d, levels=%d, factors=%s",
        len(mesh.v),
        len(mesh.f),
        len(factors),
        factors,
    )
    M, A, D, U = [], [], [], []
    A.append(get_vert_connectivity(mesh.v, mesh.f).tocoo())
    M.append(mesh)

    for i, factor in enumerate(factors):
        level_start = time.perf_counter()
        logger.info(
            "Transform level %d/%d started: current_vertices=%d, current_faces=%d, keep_fraction=%.4f",
            i + 1,
            len(factors),
            len(M[-1].v),
            len(M[-1].f),
            factor,
        )
        ds_f, ds_D = qslim_decimator_transformer(M[-1], factor=factor)
        D.append(ds_D.tocoo())
        new_mesh_v = ds_D.dot(M[-1].v)
        new_mesh = Mesh(v=new_mesh_v, f=ds_f)
        M.append(new_mesh)
        A.append(get_vert_connectivity(new_mesh.v, new_mesh.f).tocoo())
        U.append(setup_deformation_transfer(M[-1], M[-2]).tocoo())
        logger.info(
            "Transform level %d/%d finished: new_vertices=%d, new_faces=%d, elapsed=%.2fs",
            i + 1,
            len(factors),
            len(new_mesh.v),
            len(new_mesh.f),
            time.perf_counter() - level_start,
        )

    logger.info("Transform matrices generated: n_nodes=%s", [len(m.v) for m in M])
    return M, A, D, U
