class CartesianMesh:
    def __init__(
        self,
        blocks_shape: tuple,
        extent=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)),
    ):
        self._blocks_shape = blocks_shape
        self._extent = extent

    def patch_extent(self, index):
        """
        Return the box-like region covering the patch at the given location.

        If `level` is `None`, then `index` is must be a topological index,
        otherwise if `level` is an integer then `index` must be a geometrical
        index.
        """

        level, (i, j, k) = index
        (x0, x1), (y0, y1), (z0, z1) = self._extent
        dx = (x1 - x0) / (1 << level)
        dy = (y1 - y0) / (1 << level)
        dz = (z1 - z0) / (1 << level)

        return (
            (x0 + dx * i, x0 + dx * (i + 1)),
            (y0 + dy * j, y0 + dy * (j + 1)),
            (z0 + dz * k, z0 + dz * (k + 1)),
        )

    def coordinate_array(self, index, location="vert"):
        """
        Return an array of coordinates at the given location.

        If `level` is `None`, then `index` is must be a topological index,
        otherwise if `level` is an integer then `index` must be a geometrical
        index.
        """
        from numpy import linspace, meshgrid, stack

        level, (i, j, k) = index

        (x0, x1), (y0, y1), (z0, z1) = self.patch_extent(index)
        ni, nj, nk = self._blocks_shape
        dx = (x1 - x0) / ni
        dy = (y1 - y0) / nj
        dz = (z1 - z0) / nk

        if location == "vert":
            x = linspace(x0, x1, ni + 1)
            y = linspace(y0, y1, nj + 1)
            z = linspace(z0, z1, nk + 1)

        elif location == "cell":
            x = linspace(x0 + 0.5 * dx, x1 - 0.5 * dx, ni)
            y = linspace(y0 + 0.5 * dy, y1 - 0.5 * dy, nj)
            z = linspace(z0 + 0.5 * dz, z1 - 0.5 * dz, nk)

        x, y, z = meshgrid(x, y, z, indexing="ij")
        return stack((x, y, z), axis=-1)

    def cell_coordinate_array(self, *args, **kwargs):
        return self.coordinate_array(location="cell", *args, **kwargs)

    def vert_coordinate_array(self, *args, **kwargs):
        return self.coordinate_array(location="vert", *args, **kwargs)


def test_grid():
    def initial_data(xyz):
        from numpy import exp

        x = xyz[..., 0]
        y = xyz[..., 1]
        return exp(-50 * (x**2 + y**2))

    cell_coords = dict()
    vert_coords = dict()
    primitive = dict()
    level = 3
    geom = CartesianMesh(blocks_shape=(10, 10, 1))

    for i in range(1 << level):
        for j in range(1 << level):
            index = (level, (i, j, 0))
            cell_coords[index] = geom.cell_coordinate_array(index)
            vert_coords[index] = geom.vert_coordinate_array(index)
            primitive[index] = initial_data(cell_coords[index])

    from matplotlib import pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 10))

    for vert, prim in zip(vert_coords.values(), primitive.values()):
        ax1.pcolormesh(
            vert[:, :, 0, 0],
            vert[:, :, 0, 1],
            prim[:, :, 0],
            edgecolors="k",
            vmin=0.0,
            vmax=1.0,
        )

    ax1.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    test_grid()
