
import pickle
import logging
from pathlib import Path



import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ase.calculators.calculator import Calculator
from ase.io import read, write
from ase.build import bulk, make_supercell
from ase import Atoms
from ase.build import bulk
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS
from ase.build import bcc110, bcc100, bcc111
from ase.units import J, m
from pymatgen.transformations.standard_transformations import DeformStructureTransformation
from pymatgen.io.ase import AseAtomsAdaptor

from utils.general import relaxer_func, HiddenPrints


logger = logging.getLogger("master_eval")




def calc_fe_lattice_constant(calc: Calculator) -> Atoms:
    """
    Takes in a calculator and relaxes a bcc fe lattice
    and returns the equilibrium structure (Atoms object)
    """
    a_0 = 3.10

    # make a Fe bcc crystal
    atoms = bulk('Fe', 'bcc', a=a_0, cubic=True)

    # set calculator
    atoms.calc = calc

    atoms = relaxer_func(atoms, fmax=0.001, relax_cell=True)

    return atoms


def main_lattice_constant(calc: Calculator, exp_name: str) -> dict:
    """
    Function to calculate the lattice parameter of bcc Fe
    with the given Calculator. It displays the lattice parameter
    in stdout.
    Args:
        calc: Calculator object
        exp_name: name of the experiment for MLIP

    """
    logger.info("Getting equilibrium structure of Fe")
    fe_bulk = calc_fe_lattice_constant(calc)
    print("-------------------")
    print(f"Lattice constants obtained from {exp_name}")
    print(f"a = {fe_bulk.cell.cellpar()[0]}")
    print(f"b = {fe_bulk.cell.cellpar()[1]}")
    print(f"c = {fe_bulk.cell.cellpar()[2]}")
    print(f"alpha = {fe_bulk.cell.cellpar()[3]}")
    print(f"beta = {fe_bulk.cell.cellpar()[4]}")
    print(f"gamma = {fe_bulk.cell.cellpar()[5]}")

    # logging info
    logger.info(f"a = {fe_bulk.cell.cellpar()[0]}")
    logger.info(f"b = {fe_bulk.cell.cellpar()[1]}")
    logger.info(f"c = {fe_bulk.cell.cellpar()[2]}")
    logger.info(f"alpha = {fe_bulk.cell.cellpar()[3]}")
    logger.info(f"beta = {fe_bulk.cell.cellpar()[4]}")
    logger.info(f"gamma = {fe_bulk.cell.cellpar()[5]}")
    dct = { 'a' : [fe_bulk.cell.cellpar()[0]],
           'b' : [fe_bulk.cell.cellpar()[1]],
           'c' : [fe_bulk.cell.cellpar()[2]],
           'alpha' : [fe_bulk.cell.cellpar()[3]],
           'beta' : [fe_bulk.cell.cellpar()[4]],
           'gamma' : [fe_bulk.cell.cellpar()[5]],
             }
    return dct




def calculate_vacancy_energy(atoms: Atoms, calc: Calculator) -> float:
    """
    Calculate the vacancy formation energy of a 4x4x4 supercell
    Args:
        atoms: Atoms object with conventional or primitive unit cell.
              ( 4 atoms for fcc and 2 atoms for bcc)
        calc: Calculator object
    return: 
           vac_formation_energy: float  calculated vacancy formation energy
    """

    atoms.calc = calc
    # make a supercell
    logger.info(f"Making supercell([[4, 0, 0], [0, 4, 0], [0, 0, 4]]) of passed eq atoms object  ")
    super_cell = make_supercell(atoms, [[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    super_cell.calc = atoms.calc
    bulk_energy = super_cell.get_potential_energy()

    # delete one atom
    logger.info(f"Deleting atom at position 0")
    del super_cell[0]

    # relax the position of atoms without relaxing the cell
    logger.info(f"Relaxing structure with vacancy")
    vac_cell = relaxer_func(super_cell)
    vac_energy = vac_cell.get_potential_energy()

    num_atoms = len(super_cell)
    vac_formation_energy = vac_energy - \
        ((num_atoms/(num_atoms + 1)) * bulk_energy)

    print(f"{vac_formation_energy=}")
    logger.info(f"{vac_formation_energy=}")

    return vac_formation_energy


def main_vacancy_energy(calc: Calculator, exp_name: str)-> float:
    print("Calculating vacancy energy")
    logger.info("Getting Fe equilirbium structure..")
    fe_bulk = calc_fe_lattice_constant(calc)
    vac_energy = calculate_vacancy_energy(fe_bulk, calc)
    print(f"{vac_energy=}")
    return vac_energy




def relax_bulk_for_elastic_tensor(calc, f_max: float) -> Atoms:
    """
    This funciton is to relax a bcc Fe bulk to get the
    equilibrium structure.
    args: 
        calc: Calculator object of the MLIP.
        f_max: allowed maximium force acting on the atom
    """
    n_cells = 1
    a0Fe_initial = 2.8489201513672953
    # f_max = 1e-5

    # create a bulk crystal
    Pure_bulk = bulk('Fe',
                     'bcc',
                     a=a0Fe_initial,
                     cubic=True)
    multiplier = np.identity(3) * n_cells
    Pure_bulk = make_supercell(Pure_bulk, multiplier)

    # set the calculator
    Pure_bulk.calc = calc

    # Relaxation
    ecf = UnitCellFilter(Pure_bulk,
                         hydrostatic_strain=True,
                         cell_factor=float(len(Pure_bulk) * 10))
    opt = BFGS(ecf)
    opt.run(fmax=f_max, steps=100)
    return Pure_bulk

def fit_for_elastic_tensor(e, c, b):  # S=C*E
    return c * e + b

def calculate_elastic_tensor(calc: Calculator, f_max: float = 5e-5, verbose:bool = True) -> np.array:
    """
    This model takes in the Calculator object and returns the elastic tensor for BCC Fe.
    args: 
        calc: calcuator object of MLIP
        f_max: allowed maximium force acting on the atom
    """

    # get Fe relaxed bulk structure
    logger.info(f"verbose set to {verbose}")
    if verbose:
        Pure_bulk = relax_bulk_for_elastic_tensor(calc, f_max)
    else:
        print(f"verbose set to False")
        with HiddenPrints():
            Pure_bulk = relax_bulk_for_elastic_tensor(calc, f_max)

    elastic_constants_tensor = np.array(np.zeros([6, 6]))

    for column in [0, 1, 2, 3, 4, 5]:
        deformations = np.array([-0.01, -0.005, 0.005, 0.01])

        df_stress_response = pd.DataFrame(columns=['Sigma11', 'Sigma22', 'Sigma33', 'Sigma23', 'Sigma13', 'Sigma12'])

        for deformation in deformations:
            # Convert to pymatgen
            pymat = AseAtomsAdaptor.get_structure(Pure_bulk)

            # apply deformation
            if column == 0:
                Transformation_matrix = np.array([[1 + deformation, 0, 0], [0, 1, 0, ], [0, 0, 1]])
            if column == 1:
                Transformation_matrix = np.array([[1, 0, 0], [0, 1 + deformation, 0, ], [0, 0, 1]])
            if column == 2:
                Transformation_matrix = np.array([[1, 0, 0], [0, 1, 0, ], [0, 0, 1 + deformation]])
            if column == 3:
                Transformation_matrix = np.array([[1, 0, 0], [0, 1, deformation / 2], [0, deformation / 2, 1]])
            if column == 4:
                Transformation_matrix = np.array([[1, 0, deformation / 2], [0, 1, 0, ], [deformation / 2, 0, 1]])
            if column == 5:
                Transformation_matrix = np.array([[1, deformation / 2, 0], [deformation / 2, 1, 0, ], [0, 0, 1]])

            pymat_deformed = DeformStructureTransformation(Transformation_matrix).apply_transformation(pymat)

            # Convert back to ASE
            deformed_structure = AseAtomsAdaptor.get_atoms(pymat_deformed)

            # relax only atomic positions
            deformed_structure.calc = calc
            if verbose:
                opt = BFGS(deformed_structure)
                opt.run(fmax=f_max, steps=100)
            else: 
                with HiddenPrints():
                    opt = BFGS(deformed_structure)
                    opt.run(fmax=f_max, steps=100)

            # measure the stress response [eV/A^3]
            stress_response = deformed_structure.get_stress()
            # Adding the stresses from each applied deformation
            df_stress_response.loc[len(df_stress_response), df_stress_response.columns] = stress_response

        # calculating a column of the elastic constants tensor
        c_column = []
        for j, stress_component in enumerate(df_stress_response.columns):
            Copt = curve_fit(fit_for_elastic_tensor, deformations, df_stress_response[stress_component])[0]
            c_column.append(Copt)
            elastic_constants_tensor[j][column] = c_column[j][0] * m ** 3 / J / 1e9

    # np.set_printoptions(formatter={'float': "{0:0.3f}".format})
    return elastic_constants_tensor


def determine_for_calc(calc: Calculator, 
                       calc_name: str='', 
                       f_max: float = 5e-5,
                       verbose:bool = True) -> pd.DataFrame:
    list_cubic_elastic_constants = []
    Elastic_constants_tensor = calculate_elastic_tensor(calc, f_max, verbose)
    dic_cubic_elastic_constants = {'Calculator': calc_name,
                                   'C11': Elastic_constants_tensor[0][0],
                                   'C12': Elastic_constants_tensor[0][1],
                                   'C44': Elastic_constants_tensor[3][3],
                                   'B': (Elastic_constants_tensor[0][0] + 2*Elastic_constants_tensor[0][1])/3,
                                   'c_dash':(Elastic_constants_tensor[0][0] - Elastic_constants_tensor[0][1])/2,

                                   }
    list_cubic_elastic_constants.append(dic_cubic_elastic_constants)

    
    return pd.DataFrame(list_cubic_elastic_constants)





def get_energy_volume(calc: Calculator,
                         custom_pickle_path: str|None = None):
    """
    Loads pickle list of energy_volume curve and then 
    predicts the energies for the structures
    If passing path to cutsom list of structures pass it as
    named keyword argues with name 'custom_pickle_path'.
    Assumes pickle files are list of Ase.Atoms object
    Returns:
    pd.Dataframe with columns
        energies: list of energies/atom
        volumes: list of volumes
    """
    if custom_pickle_path is not None:
        with open(custom_pickle_path, 'rb') as f:
            ev_structures = pickle.load(f)
    else: 
        with open(f'input/ev_structures.pkl', 'rb') as f:
            ev_structures = pickle.load(f)

    energies = []
    volumes = []
    for structure in ev_structures:
        # get the energy of the structure
        structure.calc = calc
        energy = structure.get_potential_energy() / len(structure)
        energies.append(energy)
        volumes.append(structure.get_volume())

    dct = {'energy_per_atom': energies,
        'volume': volumes}

    return pd.DataFrame(dct)



######################################################################


def get_gb_energy(gb_pd, calc:Calculator, relax_cell: bool = False):
    """
    This function  takes gbs in a pandas dataframe and then
    calculates the GB energies. The GBs are taken from literature
    and then directly calculated. When determining the GB energy
    by default the super cell is not allowed to relax only atom
    positions are allowed to be relaxed.
    """


    # print(f"saving gbs")


    #-----------------ref prediction----------------------------------
    a_0 = 2.86
    atoms = bulk('Fe', 'bcc', a=a_0, cubic=True)

    # set calculator
    atoms.calc = calc

    relax_structure = relaxer_func(atoms, relax_cell=True)
    ref_pred_energy = relax_structure.get_potential_energy() / len(relax_structure)
    
    # if the gb_pd already has 'gb_energy' column delete it

    if 'gb_energy' in gb_pd.columns:
        gb_pd.drop(columns=['gb_energy'], inplace=True) # this is the dft values 

    #-----------------prediction for structures in dataset----------------------------------

    # iterate through each row and predict the energy
    for index, row in gb_pd.iterrows():
        
        gb_struct: Atoms = row['atoms']
        gb_struct.calc = calc
        prediction = gb_struct.get_potential_energy()

        relax_gb_structure  = relaxer_func(gb_struct, relax_cell=relax_cell)
        gb_pred_energy = relax_gb_structure.get_potential_energy()
        

        excess_energy = (gb_pred_energy - (ref_pred_energy * len(relax_gb_structure))) 
        excess_energy = excess_energy / (2 * gb_struct.cell.cellpar()[0] * gb_struct.cell.cellpar()[1])
        logger.info(f"Excess energy for {index} is {excess_energy} eV/A^2")
        logger.info(f"converting to J/mm^2")
        gb_pd.at[index, 'gb_energy'] = excess_energy * 1.60218*10**(-19)/(10**-20)


    gb_pd.drop(columns=['atoms'], inplace=True) # delete atoms column to save memory and clean logging
    logger.info("GB energies in J/mm^2")
    logger.info(gb_pd.to_string())
    return gb_pd



def save_gbs(gb_pd: pd.DataFrame, save_dir: Path):
    """
    function to save the unrelaxed GBs in the dataframe
    """

    structs = gb_pd['atoms'].values
    x_tick_label = [f'sigma-{i}_{str(j[0])}' for i,j in zip(gb_pd['sigma'], gb_pd['plane'])]
    for idx, struct in enumerate(structs):
        save_name = Path(f"{save_dir}/{x_tick_label[idx]}")
        save_name.mkdir(parents=True, exist_ok=True)
        # struct.to(f"{save_name}/unrelaxed.poscar", fmt='poscar')
        write(filename=f"{save_name}/unrelaxed.poscar", images=struct, format='vasp')


def convert_gb_structures_to_atoms():
    """
    This is a temporary function to save the pd dataframe with grain boundaries 
    (pymatgen structures object) to ase atoms object.
    """

    gb_path:Path = Path(f'/home/naveen/Ubu/code/phd/chgnet/output/default/gbs')
    gb_pd: pd.DataFrame = pd.read_pickle(Path(gb_path, 'gb_structures.pkl'))
    structures = gb_pd['structures'].values
    atoms = [AseAtomsAdaptor.get_atoms(i) for i in structures]
    gb_pd['atoms'] = atoms
    # delete a column in pandas
    del gb_pd['structures']
    gb_pd.to_pickle(Path(gb_path, 'gb_atoms.pkl'))
    print("saved as with atoms object")



def eval_surface(calc: Calculator, save_dir: Path | str | None = None) -> pd.DataFrame:
    
    if save_dir is None:
        save_dir = Path("./testing/surfaces") 
        save_dir.mkdir(exist_ok=True, parents=True)

    elif isinstance(save_dir, str):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

    elif isinstance(save_dir, Path):
        save_dir = Path(f"{save_dir}/surfaces")
        save_dir.mkdir(exist_ok=True, parents=True)

    else: 
        raise ValueError("save_dir must be a Path, str or None")

    # create bulk bcc to get the lattice parameter
    logger.info("Calculating the equilibrium structure for Fe to generate surface...") 
    bulk_fe: Atoms = bulk('Fe', 'bcc', a=2.86, cubic=True)
    bulk_fe.calc = calc 
    bulk_fe: Atoms = relaxer_func(bulk_fe, relax_cell=True)
    energy_bulk_fe_per_atom = bulk_fe.get_potential_energy() / len(bulk_fe)
    lat_par: float = bulk_fe.cell.cellpar()[0]

    logger.info(f"lat_par: {lat_par}")
    dct_surface ={}
    
    surfaces = ['bcc100', 'bcc110', 'bcc111']
    logger.info(f"Evaluating for surface {surfaces}")

    for idx, func in enumerate([bcc100, bcc110, bcc111]):
        logger.info(f"Generating surface: {surfaces[idx]}")
        # func is a callable either bcc100, bcc110, bcc111
        surface = func('Fe',size=(4,4,20), a=2.86, vacuum=10, orthogonal=True)
        logger.info(f"Writing the poscar file before relax in:  {save_dir}/{surfaces[idx]}_before_relax.poscar")
        write(filename=f"{save_dir}/{surfaces[idx]}_before_relax.poscar", images=surface, format='vasp')

        surface.calc = calc
        surface: Atoms = relaxer_func(surface, relax_cell=False)
        logger.info(f"Writing the poscar file after relax in:  {save_dir}/{surfaces[idx]}_after_relax.poscar")
        write(filename=f"{save_dir}/{surfaces[idx]}_after_relax.poscar", images=surface, format='vasp')

        (a, b)= (surface.cell.cellpar()[0], surface.cell.cellpar()[1])
        logger.info(f"Surface dimensions a,b: {a}, {b}")

        energy_surface = surface.get_potential_energy()
        s_energy = (energy_surface - ( len(surface) * energy_bulk_fe_per_atom)) / (2*a*b)

        dct_surface[surfaces[idx]] = [s_energy]
        logger.info(f"Surface: {surfaces[idx]} Energy: {s_energy}")

    logger.info(pd.DataFrame(dct_surface).to_string())

    return pd.DataFrame(dct_surface)
