
from __future__ import annotations
from typing import TYPE_CHECKING
from datetime import date, datetime
import logging
import sys
import os
from pathlib import Path

import torch
from ase.optimize import  FIRE
from ase.filters import StrainFilter, FrechetCellFilter

import yaml
from pathlib import Path
from dataclasses import dataclass

if TYPE_CHECKING:
    from ase.atoms import Atoms


# DEVICE = 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VERBOSE = False # only used to print the steps in FIRE during relaxation

# logger = logging.getLogger(__name__)
logger = logging.getLogger("master_eval")

class HiddenPrints:
    """
    This is to suppressing printing
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


@dataclass
class Exp_info():
    """
    A data class to load related info for a model.
    
    calc_name: can be any name, the results will be stored in dir with calc_name in the output dir.
    calc_path: Used to load the model from the given path
    """
    calc_name: str
    calc_path: str
    parent_dir_path: str  # only needed for plotting
    

def load_info(info_file: Path)-> list[Exp_info]:
    """
    Loads the yaml file and returns list of calcs to
    evaluate
    """
    if info_file.exists():

        with open(info_file, 'r') as f:
            data: dict = yaml.safe_load(f)

        models = []
        for key, value in data.items():
            models.append(Exp_info(**value))

        return models
    else:
        raise FileNotFoundError(f"{info_file} Yaml file not found. Run create_template to create a template")


def create_template(info_file:Path):
    """
    Creates a template for the info file
    """

    data = {
            1: {"calc_name": "calc_name",
                "calc_path": "calc_path",
                "parent_dir_path": "parent_dir_name"},
            2: {"calc_name": "second calc",
                "calc_path": "second_path",
                "parent_dir_path": "parent_dir_name"},

            }

    with open(info_file, 'w') as f:
        yaml.safe_dump(data, f)



def setup_logger(log_prefix: str, logger_name: str = __name__):
    """
    Sets up a new logger and saves in the logs/{current_date_time}_{log_prefix}
    """

    def handle_file_exists(log_file: str, count: int =0):
        if isinstance(log_file, Path):
            log_file = str(log_file)

        if Path(log_file).exists():
            log_file = log_file.strip(f".{count-1}")
            log_file = log_file + f".{count}"  # Rename existing log file to .old
            log_file = handle_file_exists(log_file, count + 1 )  # Create a new log file with .log suffix
    
        return log_file
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    set_log_level = logging.INFO  # Set the log level to DEBUG for detailed logs
    # Get current date for log file name
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file = f"{log_dir}/{current_date}_{log_prefix}.log"
    log_file = handle_file_exists(log_file)

    print(f"Using log file: {log_file}")
    
    # Get or create logger
    # logger = logging.getLogger(__name__)
    logger = logging.getLogger(logger_name)
    
    # Prevent adding handlers if logger is already configured
    if not logger.handlers:
        logger.setLevel(set_log_level)
        
        # Create file handler (logs only to file, no stdout)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(set_log_level)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        
        # Add only the file handler to logger
        logger.addHandler(file_handler)
        
        # Ensure no propagation to root logger (which may have a StreamHandler)
        logger.propagate = False
    
    return logger


def relaxer_func(atoms: Atoms,
                 fmax: float = 0.001,
                 relax_cell: bool = False,
                 force_symmetry: bool = False,
                 steps: int=2000, 
                 verbose: bool = False) -> Atoms:
    """
    function to relax the structure using FIRE algorithm.
    Args:
        atoms (Ase.Atoms) : Atoms object to be relaxed with the calculator attached to it.
        fmax (float): maximum force tolerance for relaxation. Default is 0.001 eV/Ang
        relax_cell (bool): whether to relax the cell parameters or not. Default is
                    False.
        force_symmetry (bool): whether to enforce symmetry during cell relaxation or
                        not. Default is False.
        steps (int): The max number of iterations if not converged. Default is 2000.
        verbose (bool): whether to print the output of optimizer or not. Default is False.
    """

    logger.debug(f"{relax_cell=}")
    if relax_cell:
        logger.info("Relaxing cell parameters")
        if force_symmetry:
            logger.info("preserving symmetry during cell relaxation using FrechetCellFilter with mask 1,1,1,0,0,0")
            # to also relax the cell parameters. preserves symmetry
            # atoms = ExpCellFilter(atoms)
            # ExpCellFilter is deprecated so Frechet filter is used
            # 
            atoms = FrechetCellFilter(atoms, mask=[1,1,1,0,0,0])
        else:
            logger.info("Not preserving symmetry during cell relaxation using StrainFilter")
            # to also relax the cell parameters. does not enforce symmetry
            atoms = StrainFilter(atoms)
    max_steps = None
    n_steps = None
    if verbose or VERBOSE:
        dyn = FIRE(atoms)
        dyn.run(fmax=fmax, steps=steps)
        max_steps = dyn.max_steps
        n_steps = dyn.nsteps
        print(f"{dyn.max_steps=}")
        print(f"{dyn.nsteps=}")

    else:
        with HiddenPrints():
            dyn = FIRE(atoms)
            dyn.run(fmax=fmax, steps=steps)
        max_steps = dyn.max_steps
        n_steps = dyn.nsteps

    if max_steps == n_steps:
        logger.info("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        logger.info("Not converged")
        logger.info("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print("xxxxxxxxxxxxxxx NOT CONVERGED xxxxxxxxxxxxxxx")

    else: 
        logger.debug("CONVERGED")
        logger.debug(f"{dyn.max_steps=}")
        logger.debug(f"{dyn.nsteps=}")

    if relax_cell:
        return atoms.atoms

    return atoms



