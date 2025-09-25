"""
This is to evalaute grain boundary segregation
"""
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.atoms import Atoms
from ase.build import make_supercell
from utils.eval_fe_bulk import calc_fe_lattice_constant
from utils.eval_fe_substitutional import get_fe_x_energy
from utils.general import relaxer_func


def determine_voronoi_volume(struct: Atoms, reps=1):
    """
    returns the voronoi volume of each atom
    uses scipy to detetmine the voronoi volumes 
    """
    assert isinstance(struct, Atoms), "Input must be an ASE Atoms object"
    from scipy.spatial import Voronoi, ConvexHull
    cell = struct.get_cell()
    positions = struct.get_positions()
    n_atoms = len(struct)

    images = []
    shifts = range(-reps, reps+1)
    for i in shifts:
        for j in shifts:
            for k in shifts:
                shift = i * cell[0] + j* cell[1] + k*cell[2]
                images.append(positions+shift)

    all_points = np.vstack(images)
    
    # run voronoi
    vor = Voronoi(all_points)

    volumes = np.zeros(n_atoms)

    for idx in range(n_atoms):
        region_index = vor.point_region[idx]
        vertices = vor.regions[region_index]

        if -1 in vertices or len(vertices) == 0:  # some regions can be open
            volumes[idx] = np.inf
            continue

        polyhedron = vor.vertices[vertices]
        try:
            hull = ConvexHull(polyhedron)
            volumes[idx] = hull.volume
        except Exception:
            volumes[idx] = 0.0

    return volumes



def calc_gb_segregation_energy(e_fe_x_gb,e_fe_gb,e_fe_x, e_fe) :
    """
    Equation to calculate the gb segregation energy
    e_fe_x_gb -> eneryg of Fe grain boundary supercell with one impurity at the GB M-1 Fe atoms
    e_fe_gb -> energy of Fe grain boundary supercell M Fe Atoms
    e_fe_x -> energy of Fe bulk supercell with one impurity  N-1 Fe atoms
    e_fe  -> energy of Fe bulk supercell with N Fe atoms
    """
    
    return ((e_fe_x_gb - e_fe_gb) - (e_fe_x - e_fe))


def get_fe_x_gb_energy(gb, imp, calc, save_dir: Path|None=None):
    """
    introduces impurity at each site of the GB and then
    determines the energy of the supercell
    Save_dir can be used to save the relaxed gb structure with impurity
    """
    num_atoms_in_gb = len(gb)

    for i in range(num_atoms_in_gb):
        gb_with_imp = gb.copy()

        gb_with_imp[i].symbol = imp
        gb_with_imp.calc = calc
        gb_with_imp = relaxer_func(gb_with_imp) # relax the gb with impurity

        if save_dir is not None:
            Path(f"{save_dir}/gb_with_{imp}").mkdir(parents=True, exist_ok=True)
            write(filename=f"{save_dir}/gb_with_{imp}/gb_with_{imp}_at_{i}.xyz", images=gb_with_imp, format='extxyz')

        yield gb_with_imp.get_potential_energy()


def get_gb_segregation_energy(gb: Atoms, imp: str, calc: "Calculator",
                              save_dir:Path|None=None)-> Atoms:
    """
    Calculates the segregation energy of one solute atom in the various
    sites of the GB. Saves a Fe atoms object with each site containing
    energy corresponding to the segregation energy.
    """

    # get energy of the Fe bulk
    fe_eq = calc_fe_lattice_constant(calc)

    fe_supercell = make_supercell(fe_eq, [[4, 0, 0], [0, 4, 0], [0, 0, 4]])

    fe_supercell.calc = calc
    e_fe_supercell = fe_supercell.get_potential_energy()

    #  get the fe with impurity in bulk energy
    e_fe_x_supercell = get_fe_x_energy(fe_supercell, imp, calc)

    # relax gb with positions of atoms and get energy
    gb.calc = calc
    relaxed_gb = relaxer_func(gb)
    e_fe_gb = relaxed_gb.get_potential_energy()

    seg_energies_per_site = []

    for e_fe_x_gb in get_fe_x_gb_energy(relaxed_gb, imp, calc, save_dir):
        # get the segregation energy
        seg_energy = calc_gb_segregation_energy(e_fe_x_gb=e_fe_x_gb, 
                                                e_fe_gb=e_fe_gb,
                                                e_fe_x=e_fe_x_supercell,
                                                e_fe=e_fe_supercell)
        seg_energies_per_site.append(seg_energy)
    
    # introduce the impurity at all sites
    relaxed_gb.arrays['segregation_energies'] = np.asarray(seg_energies_per_site)

    return relaxed_gb


def test_per_gb(calc, imps: list, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    for imp in imps:

        gb_supercell_imp = get_gb_segregation_energy(gb_supercell,  imp=imp, calc=calc,
                                                  save_dir=save_dir)

        write(filename=f"{save_dir}/final_gb_{imp}.xyz", images=gb_supercell_imp, format='extxyz')


def test_gb(calc, save_dir):
    """
    tests the function
    """
    # use a arbitrary gb
    # gb = None
    save_dir.mkdir(parents=True, exist_ok=True)

    gbs = read("input/fe_gbs.xyz", index=":")
    imps = ["Cu", "Ni", "Sn"]

    for gb in gbs:
        gb_dir_name = 'sigma'

        for key in ['sigma', 'type', 'plane']:
            gb_dir_name += "_"+str(gb.info[key]).strip("'[]")
        gb_supercell = make_supercell(gb, [[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        test_per_gb(calc, imps, save_dir / gb_dir_name)



if __name__ == "__main__":
    # test_gb()
    # vol = determine_voronoi_volume(read("../input/test_10.xyz", index=":")[0])
    # print(vol)
    # for i, v in enumerate(vol):
    #     print(f"Atoms{i}: Voronoi volume = {v}")
