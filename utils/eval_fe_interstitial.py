from pathlib import Path
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from ase.calculators.calculator import Calculator


import numpy as np
from ase.io import write
from ase.build import make_supercell

from utils.general import relaxer_func
from utils.eval_fe_bulk import calc_fe_lattice_constant


TESTING = False

logger = logging.getLogger("master_eval")


def eval_interstitials(imp: str, calc: Calculator, save_dir: Path):
    """
    Takes in imp, and calc
    Returns the energy of e_oct, e_tet, e_subst
    """
    save_dir = save_dir / "interstitial"
    save_dir.mkdir(exist_ok=True, parents=True)

    print("----------------------------------------")
    print("Impurity: ", imp)
    print("----------------------------------------")
    fe_bulk = calc_fe_lattice_constant(calc)
    fe_bulk.calc = calc
    e_pure = fe_bulk.get_potential_energy()/len(fe_bulk)
    print(f" Pure Fe 1 atom: {e_pure}")
    # make a supercell
    super_cell = make_supercell(fe_bulk, [[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    super_cell.calc = fe_bulk.calc

    super_cell[0].symbol = imp

    write(filename=f"{save_dir}/unrelax_{imp}_subst.poscar",images= super_cell, format='vasp')
    super_cell = relaxer_func(super_cell, fmax=0.001, relax_cell=False)

    e_sub = super_cell.get_potential_energy()
    print(f" Substitutional: {e_sub}")
    write(filename=f"{save_dir}/relax_{imp}_subst.poscar",images= super_cell, format='vasp')

    supercell_int = make_supercell(fe_bulk, [[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    solute_position = [0.5, 0.5, 0]
    absolute = np.dot(solute_position, supercell_int.cell) / 4
    supercell_int.append(imp)
    supercell_int.positions[-1] = absolute
    supercell_int.calc = fe_bulk.calc

    write(filename=f"{save_dir}/unrelax_{imp}_int_oct.poscar",images= supercell_int, format='vasp')
    supercell_int = relaxer_func(supercell_int, fmax=0.001, relax_cell=False)

    e_oct = supercell_int.get_potential_energy()
    print(f" interstitial oct: {e_oct}")
    write(filename=f"{save_dir}/relax_{imp}_int_oct.poscar",images= supercell_int, format='vasp')

    supercell_int = make_supercell(fe_bulk, [[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    solute_position = [0.5, 0.25, 0]
    absolute = np.dot(solute_position, supercell_int.cell) / 4
    supercell_int.append(imp)
    supercell_int.positions[-1] = absolute
    supercell_int.calc = fe_bulk.calc
    write(filename=f"{save_dir}/unrelax_{imp}_int_tet.poscar",images= supercell_int, format='vasp')
    supercell_int = relaxer_func(supercell_int, fmax=0.001, relax_cell=False)

    e_tet = supercell_int.get_potential_energy()
    print(f" interstitial tet: {e_tet}")
    write(filename=f"{save_dir}/relax_{imp}_int_tet.poscar",images= supercell_int, format='vasp')

    print(f"{e_oct - e_tet = }")
    e_diff_oct_tet = e_sub + e_pure - e_tet
    print(f"e_sub - e_tet = {e_diff_oct_tet} ")
    e_diff = e_sub + e_pure - e_oct
    print(f"e_sub - e_oct = {e_diff}")

    return e_oct, e_tet, e_sub


