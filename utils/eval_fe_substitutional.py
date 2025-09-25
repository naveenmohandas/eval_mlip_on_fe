import logging
from pathlib import Path



import pandas as pd

from ase.calculators.calculator import Calculator
from ase.io import read, write
from ase.build import bulk, make_supercell
from ase import Atoms
from ase.build import bulk
from ase.build import make_supercell

from utils.general import relaxer_func

logger = logging.getLogger("master_eval")


def get_impurity_energy(imp: str,
                        calc: Calculator,
                        per_atom: bool = True) -> float:
    """
    Function for obtaining the bulk energy of impurity atom.
    Args:
        imp: impurity atom
        calc: Calculator object
        per_atom: whether to return energy per atom or not. Default is True.
    returns: float
        impurity energy per atom or total energy
    """

    # load the impurity structure
    file_path = Path(
        f"./input/poscar-structures/equilibrium-structures/{imp}-POSCAR")
    logger.debug(f"Loading {file_path=}")
    imp_structure: Atoms = read(file_path)
    imp_structure.calc = calc

    if per_atom:
        logger.info("returning energy per atom for impurity energy")
        return imp_structure.get_potential_energy()/len(imp_structure)
    else:
        logger.info(f"returning total energy")
        return imp_structure.get_potential_energy()


def get_fe_x_energy(fe_bulk: Atoms, imp:str, calc: Calculator | None = None) -> float:
    """
    Function to obtain the energy of Fe-X system. Relax cell is set 
    to False.
    Args: 
        fe_x: Atoms object with Fe-X system. X is one impurity atom.
        calc: Calculator object
    returns: float
        total energy of the fe-x system
    """

    fe_x = fe_bulk.copy()
    fe_x.calc = calc

    fe_x[0].symbol = imp

    logger.info("Calculating the energy for Fe-x system")
    fe_x = relaxer_func(fe_x, fmax=0.001, relax_cell=False)

    return fe_x.get_potential_energy()


def calc_binding_energy(fe_x_y: Atoms, calc: Calculator, **kwargs) -> tuple[float, Atoms]:
    """
    Function to calcualte the binding energy of Fe with impurity X and Y.
    Here also the relax cell is set to False for Fe-X-Y system.
    Args:
        fe_x_y: Atoms object with Fe-X-Y system
        calc: Calculator object
        kwargs:
               fmax: force tolerance for relaxation. Default is 0.001
    """

    if 'fmax' in kwargs:
        fmax = kwargs['fmax']
    else:
        fmax = 0.001

    logger.info(f"Making Fe bulk structure for calculating binding energy")
    fe_bulk = bulk('Fe', 'bcc', a=2.9, cubic=True)
    fe_bulk = make_supercell(fe_bulk, [[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    fe_bulk.calc = calc
    fe_bulk = relaxer_func(fe_bulk, fmax=fmax, relax_cell=True)
    fe_bulk_energy = fe_bulk.get_potential_energy()
    logger.debug(f"{fe_bulk_energy=}")

    imp_x = [ele for ele in fe_x_y.get_chemical_symbols() if ele != 'Fe'][0]
    logger.info(f"{imp_x=}")
    imp_y = [ele for ele in fe_x_y.get_chemical_symbols() if ele != 'Fe'][1]
    logger.info(f"{imp_y=}")

    # replace Fe_bulk with first of the impurity
    fe_x = fe_bulk.copy()
    logger.debug(f"Replacing atom 0 with impurity {imp_x}")
    fe_x[0].symbol = imp_x
    fe_x_energy = get_fe_x_energy(fe_x, calc)
    logger.debug(f"{fe_x_energy=}")

    # replace Fe_bulk with second of the impurity
    fe_y = fe_bulk.copy()
    logger.debug(f"Replacing atom 0 with impurity {imp_y}")
    fe_y[0].symbol = imp_y
    fe_y_energy = get_fe_x_energy(fe_y, calc)
    num_atoms = len(fe_x_y)
    logger.debug(f"{fe_y_energy=}")

    fe_x_y.calc = calc
    fe_x_y = relaxer_func(fe_x_y, fmax=fmax, relax_cell=False)
    fe_x_y_energy = fe_x_y.get_potential_energy()
    logger.debug(f"{fe_x_y_energy=}")

    binding_energy = (fe_x_y_energy
                      + (fe_bulk_energy)
                      - (fe_x_energy)
                      - (fe_y_energy)
                      )
    logger.debug(f"{binding_energy=}")

    return binding_energy, fe_x_y


def calc_binding_energy_vac_vac(fe_x_y: Atoms, calc: Calculator, **kwargs) -> tuple[float, Atoms]:
    logger.info("Calculating binding energy for vacancy-vacancy")

    if 'fmax' in kwargs:
        fmax = kwargs['fmax']
    else:
        fmax = 0.001

    logger.debug(f"Making Fe bulk structure with 4x4x4 supercell for Fe-vac-vac")
    fe_bulk = bulk('Fe', 'bcc', a=2.9, cubic=True)
    fe_bulk = make_supercell(fe_bulk, [[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    fe_bulk.calc = calc
    fe_bulk = relaxer_func(fe_bulk, fmax=fmax, relax_cell=True)
    fe_bulk_energy = fe_bulk.get_potential_energy()
    logger.debug(f"{fe_bulk_energy=}")

    # get the vacancy energy
    logger.debug(f"deleting atom 0")
    fe_y = fe_bulk.copy()
    del fe_y[0]
    fe_y_energy = get_fe_x_energy(fe_y, calc)
    logger.debug(f"energy with vacancy {fe_y_energy=}")

    fe_x_y.calc = calc
    fe_x_y = relaxer_func(fe_x_y, fmax=fmax, relax_cell=False)
    fe_x_y_energy = fe_x_y.get_potential_energy()
    logger.debug(f"energy of supercel with 2 vacancy {fe_x_y_energy=}")

    binding_energy = (fe_x_y_energy
                      + fe_bulk_energy
                      - (2*fe_y_energy))
    logger.info(f"{binding_energy=}")

    return binding_energy, fe_x_y


def calc_binding_energy_vac(fe_x_y: Atoms, calc: Calculator, **kwargs) -> tuple[float, Atoms]:
    logger.info("Calculating binding energy for vacancy")

    if 'fmax' in kwargs:
        fmax = kwargs['fmax']
    else:
        fmax = 0.001

    logger.debug(f"Making Fe bulk structure with 4x4x4 supercell for binding energy Fe-x-vac")
    fe_bulk = bulk('Fe', 'bcc', a=2.9, cubic=True)
    fe_bulk = make_supercell(fe_bulk, [[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    fe_bulk.calc = calc
    fe_bulk = relaxer_func(fe_bulk, fmax=fmax, relax_cell=True)
    fe_bulk_energy = fe_bulk.get_potential_energy()
    logger.debug(f"energy of bulk Fe {fe_bulk_energy=}")

    imp_x = [ele for ele in fe_x_y.get_chemical_symbols() if ele != 'Fe'][0]

    # replace Fe_bulk with first of the impurity
    fe_x = fe_bulk.copy()
    logger.debug(f"Replacing atom 0 with impurity {imp_x}")
    fe_x[0].symbol = imp_x
    fe_x_energy = get_fe_x_energy(fe_x, calc)
    logger.debug(f"{fe_x_energy=}")

    # get the vacancy energy
    logger.debug("Deleting atom 0 to introduce vacancy")
    fe_y = fe_bulk.copy()
    del fe_y[0]
    fe_y_energy = get_fe_x_energy(fe_y, calc)
    logger.debug(f"energy with vacancy {fe_y_energy=}")

    fe_x_y.calc = calc
    fe_x_y = relaxer_func(fe_x_y, fmax=fmax, relax_cell=False)
    fe_x_y_energy = fe_x_y.get_potential_energy()
    logger.debug(f"energy of  structure with x and vac {fe_x_y_energy=}")
    num_atoms = len(fe_x_y)

    binding_energy = (fe_x_y_energy
                      + ((num_atoms+1) * fe_bulk_energy
                         / len(fe_bulk))
                      - ((num_atoms+1)*fe_x_energy /
                         len(fe_x))
                      - (fe_y_energy))

    return binding_energy, fe_x_y


def introduce_second_impurity(fe_bulk: Atoms,
                              imp_x: str,
                              imp_y: str,
                              req_neighs: int = 5) -> list:
    """
    Function to introduce second impurity atoms and save as a poscar file
    Args:
        fe_bulk: bulk Fe structure
        imp_x: first impurity atom
        imp_y: second impurity atom
        req_neighs: neighbour number upto which second impurity
                    atom is to be introduced. Defaults to 5
    returns:
        list of Atoms objects with imp_x and imp_y [1nn, 2nn, 3nn, 4nn, 5nn]
    """
    # get distances of all atoms
    distances = ([round(fe_bulk.get_distance(0, atom, mic=True), 4)
                 for atom in range(len(fe_bulk))])
    # get unique distances
    nearest_neighbour = list(set(distances))

    # sort the distances
    nearest_neighbour.sort()

    # get index of required nearest neighbours
    nn_index = [distances.index(nn)
                for nn in nearest_neighbour[1:req_neighs+1]]

    imp_atoms = []

    for nn in nn_index:
        x_copy = fe_bulk.copy()
        if imp_y == 'vac':
            if imp_x == 'vac':
                # first delete the nn then 0
                del x_copy[nn]
                del x_copy[0]
            else:
                # first replace then delete
                x_copy[nn].symbol = imp_x
                del x_copy[0]
        else:
            x_copy[0].symbol = imp_x
            x_copy[nn].symbol = imp_y
        imp_atoms.append(x_copy)

    return imp_atoms


def get_fe_xy_nn(imp_x_y: str, calc: Calculator, fmax: float = 0.001) -> list:
    """
    This is to get the list of structures with nearest neighbours of impurity.
    in order 1nn, 2nn, 3nn, 4nn, 5nn.
    """
    imp_x = imp_x_y.split('-')[0]
    imp_y = imp_x_y.split('-')[1]
    logger.debug("Making Fe bulk with 4x4x4 supercell for introducing first and second impurity")
    fe_bulk = bulk('Fe', 'bcc', a=2.9, cubic=True)
    fe_bulk = make_supercell(fe_bulk, [[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    fe_bulk.calc = calc
    # introduce impurities in the eq structures
    fe_bulk = relaxer_func(fe_bulk, fmax=fmax, relax_cell=True)
    fe_x_y: list = introduce_second_impurity(fe_bulk, imp_x, imp_y)
    return fe_x_y


def main_binding_energy(calc: Calculator,
                          exp_name: str,
                          imp_xy_list: list , **kwargs) -> pd.DataFrame:
    """
    main function to calculate the binind energy
    args:
        calc: Calculator object
        exp_name: name of the experiment
        save_csv: whether to save the csv file or not
        imp_xy_list: list of impurity pairs to calculate the binding energy
    """

    # set fmax for relaxation
    if 'fmax' in kwargs:
        fmax = kwargs['fmax']
    else:
        fmax = 0.005  # when 0.001 is set does not converge for some calculations
    logger.info(f"force tolerance for relaxation is set to {fmax=}")

    dct = {
        "binding energy": [],
        "nn": [],
        "x-y": [],
    }

    logger.info(f"Testing for elements {imp_xy_list=}")
    for idx, imp_x_y in enumerate(imp_xy_list):
        # get list of structures with nearest neighbours of impurity
        fe_nn: list = get_fe_xy_nn(imp_x_y, calc, fmax=fmax)

        # iterate through the nns and calculate the binding energy
        for nn, fe_x_y in enumerate(fe_nn):

            print(f"{idx}: Determining BE of {imp_x_y=} {nn=}")
            logger.info(f"Determining BE of {imp_x_y=} {nn=}")
            if 'vac' in imp_x_y:
                if 'vac-vac' in imp_x_y:
                    logger.info("Calculating binding energy for vac-vac")
                    binding_energy, fe_xy_relax = calc_binding_energy_vac_vac(fe_x_y.copy(),
                                                                 calc,
                                                                 fmax=fmax)
                else:
                    logger.info("Calculating binding energy for x-vac")
                    binding_energy, fe_xy_relax = calc_binding_energy_vac(fe_x_y.copy(),
                                                             calc,
                                                             fmax=fmax) 
            else:
                logger.info("Calculating binding energy for x-y")
                binding_energy, fe_xy_relax = calc_binding_energy(fe_x_y.copy(),
                                                     calc,
                                                     fmax=fmax)

            if 'save_dir' in kwargs:
                save_dir = kwargs['save_dir'] / 'fe-xy'
                save_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Saving the relaxed and unrelaxed structures")
                write(f'{save_dir}/{exp_name}-{imp_x_y}-nn-{nn+1}.unrelax', fe_x_y, format='vasp')
                write(f'{save_dir}/{exp_name}-{imp_x_y}-nn-{nn+1}.relax', fe_xy_relax, format='vasp')
            logger.info(f"{binding_energy=}")
            print(f"BE is {binding_energy=}")
            dct["binding energy"].append(binding_energy)
            dct["nn"].append(nn+1)
            dct["x-y"].append(imp_x_y)

        logger.info("END binding energy")
        logger.info("------------------------------------------")

    logger.info("------------------------------------------")
    logger.info(pd.DataFrame(dct).to_string())
    logger.info("------------------------------------------\n\n")

    return pd.DataFrame(dct)
