"""
This is to predict energies for an entire database
if none isprovided it loads the mptrj dataset
better to run it separately as it often takes a long
time to run.
"""

import sys
import pickle
from pathlib import Path
import argparse
import logging


import pandas as pd
import numpy as np
from ase.io import read
from ase.calculators.calculator import Calculator
from utils.general import Exp_info, load_info, setup_logger


logger = logging.getLogger("predict_energy")


def load_db(db_path: Path):
    if db_path.exists():
        print("Loading DB from:", db_path)
        db = read(db_path, index=":")
    else:
        print("Couldn't find DB at the given path", db_path)
        raise FileNotFoundError

    return db


def predict_per_structure(struct,  calc:Calculator, dft_energy_key: str,
                          is_dft_energy_per_atom:bool,
                          dft_force_key: str = 'force',
                          idx=-1)-> dict | None:
    """
    This function predicts per structure
    if the dft energy is per atom then set is_dft_energy_per_atom to True otherwise False
    """

    try:
        if  is_dft_energy_per_atom:
            logger.debug(f"{is_dft_energy_per_atom = }")
            dft_energy_per_atom = struct.info[dft_energy_key]
            dft_energy = dft_energy_per_atom * len(struct)
            logger.debug(f"{len(struct) = }")
            logger.debug(f"{dft_energy_per_atom = }")
            logger.debug(f"{dft_energy = }")
        else:
            dft_energy = struct.info[dft_energy_key]
            dft_energy_per_atom = dft_energy / len(struct)

        if dft_force_key in struct.arrays:
            dft_forces = struct.arrays[dft_force_key]
            dft_force = [np.linalg.norm(force) for force in dft_forces]
        else:
            dft_force = None

        struct.calc = calc
        mlip_energy = struct.get_potential_energy()
        mlip_energy_per_atom = mlip_energy / len(struct)
        mlip_forces = struct.get_forces()
        mlip_force = [np.linalg.norm(force) for force in mlip_forces]

    except Exception as e:
        print(f"Error in structure {idx}: {e}")
        return None 

    result =  {
            "dft_energy":dft_energy,
            "dft_energy_per_atom": dft_energy_per_atom,
            "dft_force":dft_force,
            "mlip_energy":mlip_energy,
            "mlip_energy_per_atom":mlip_energy_per_atom,
            "mlip_force": mlip_force}
    return result


def save_results(dct: dict, save_dir: Path, db_name: str = "_", 
                 idx: None | int=None):
    """
    Save the results as a dictionary
    idx is used to save intermediate stages when the
    dataset is large
    """
    logger = logging.getLogger("predict_energy")
    if idx is not None:
        logger.info(f"Saving intermediate results {idx}: {save_dir}/intermediate_energy_predict/{db_name}_mlip_dft_energy_{idx}.pkl ")
        Path(f"{save_dir}/intermediate_energy_predict").mkdir(exist_ok=True, parents=True)
        with open(f"{save_dir}/intermediate_energy_predict/{db_name}_mlip_dft_energy_{idx}.pkl", 'wb') as f:
            pickle.dump(dct, f)
    else:
        logger.info(f"Saving the final result: {save_dir}/{db_name}_mlip_dft_energy.pkl")
        save_dir.mkdir(exist_ok=True, parents=True)
        with open(f"{save_dir}/{db_name}_mlip_dft_energy.pkl", 'wb') as f:
            pickle.dump(dct,f)
        print("Saved the final result....")



def predict_for_db(run_info: dict, save_frequency: int = 20000, verbose_frequency: int = 500)-> None:
    """
    run_info must contain the following information.
        db_path: Path to the database
        calc: The calculator to use
        calc_name: The name of the calculator
        save_dir: The directory to save the results
        dft_key: The key to get dft energy(total energy and not energy per atom) 
                information from atoms object
    """
    log_run_info(run_info)

    db_path = run_info["db_path"]
    calc = run_info["calc"]
    calc_name = run_info["calc_name"]
    save_dir = run_info["save_dir"]
    dft_energy_key = run_info["dft_key"]
    is_dft_energy_per_atom = run_info["is_dft_energy_per_atom"]

    # comment out towards the end 
    # TOREMOVE or have better way to check
    if is_dft_energy_per_atom:
        assert dft_energy_key != 'uncorrected_total_energy'

    db = load_db(db_path)
    
    dft_energies, dft_energies_per_atom, mlip_energies, mlip_energies_per_atom = [], [], [], []
    dft_forces, mlip_forces = [], []

    print(f"Starting the predicitons")
    for idx, struct in enumerate(db):
        dct = predict_per_structure(struct, calc, dft_energy_key,
                                    is_dft_energy_per_atom, idx=idx)
        if dct is not None:
            dft_energies.append(dct['dft_energy'])
            dft_energies_per_atom.append(dct['dft_energy_per_atom'])
            mlip_energies.append(dct['mlip_energy'])
            mlip_energies_per_atom.append(dct['mlip_energy_per_atom'])

            if dct['dft_force'] is not None:
                dft_forces.extend(dct['dft_force'])
                mlip_forces.extend(dct['mlip_force'])

            # save intermediate results
            if ((idx+1) % save_frequency) == 0:
                save_results({
                    "dft_energy": dft_energies,
                    "dft_energy_per_atom": dft_energies_per_atom,
                    "mlip_energy": mlip_energies,
                    "mlip_energy_per_atom": mlip_energies_per_atom
                }, save_dir,db_name=run_info["db_name"], idx=idx)

            # print the number of structures processed
            if ((idx+1) % verbose_frequency) == 0:
                logger.info(f"Processed for {idx} structures")

    db_name = run_info["db_name"]
    # Save the final results
    save_results({
        "dft_energy": dft_energies,
        "dft_energy_per_atom": dft_energies_per_atom,
        "mlip_energy": mlip_energies,
        "mlip_energy_per_atom": mlip_energies_per_atom,
        "dft_force": dft_forces,
        "mlip_force": mlip_forces
    }, save_dir, db_name = db_name)


def calc_chgnet(data):
    from chgnet.model import CHGNetCalculator, CHGNet

    if data.calc_name.lower() == "chg2":
        model = CHGNet.load(model_name = '0.2.0', use_device='cpu')

    elif data.calc_name.lower() == "chg3":
        model = CHGNet.load(model_name = '0.3.0', use_device='cpu')

    else:
        model_path = Path(data.calc_path)
        logger.info(f"Loading model CHGNet model: {model_path}")
        model = CHGNet.from_file(model_path)

    calc = CHGNetCalculator(model, use_device='cpu')

    return calc


def calc_mace(data):
    from mace.calculators import mace_mp

    model_path = Path(data.calc_path)
    calc = mace_mp(model=model_path, default_dtype='float64', device='cpu')
    return calc


def calc_sevenn(data):
    from sevenn.sevennet_calculator import SevenNetCalculator
    model_name = data.calc_name
    if model_name =='Sevenn':
        model_path = data.calc_path
        calc = SevenNetCalculator(data.calc_path, device='cpu')  # 7net-0, SevenNet-0, 7net-0_22May2024, 7net-0_11July2024 ...
        parent_model_dir = None

    else:
        model_path = Path(data.calc_path)
        calc = SevenNetCalculator(model_path, device='cpu')  # 7net-0, SevenNet-0, 7net-0_22May2024, 7net-0_11July2024 ...
        parent_model_dir = Path(data.parent_dir_path) if data.parent_dir_path else None
    return calc


def get_calc(model_name:str, data:Exp_info):
    """
    Function to get the calculator
    """
    
    model_name = model_name.lower()

    if model_name == "chgnet":
        calc = calc_chgnet(data)
    elif model_name == "mace":
        calc = calc_mace(data)
    elif model_name == "sevenn":
        calc = calc_sevenn(data)
    else:
        raise ValueError(f"Model {model_name} is not supported. Please implement function to call calculator. Supported ones are mace, chgnet, sevenn.")


    return calc


def test_mlip(args):
    """
    This uses the calculator to predict the eneriges of db. Loads the models and
    iterates through them.
    """
    model_name: str = args.model_type.lower()
    logger = setup_logger(log_prefix=f"{args.db_name}_{model_name}_predict_energy", logger_name="predict_energy")

    logger.info(f"Running for model {model_name}")

    db_path = Path(args.db_path)
    db_name = args.db_name
    dft_key = args.dft_energy_key
    is_dft_energy_per_atom = args.is_dft_energy_per_atom 

    print("Loading the models to eval...")
    logger.info("Loading the models to eval...")

    yaml_file = Path(f"exp_info_{model_name}.yaml") if args.yaml_file is None else Path(args.yaml_file)
    if not yaml_file.exists():
        print(f"The given yaml file not found. Given yaml file: {yaml_file}.")
        raise FileNotFoundError(f"The given yaml file not found. Given yaml file: {yaml_file}.")

    models: list[Exp_info] = load_info(yaml_file)

    print(f"Got {len(models)} models to run")

    logger.info(f"Evaluating the folowing {len(models)} models")
    for data in models:
        logger.info(f"\t{data.calc_name}")
        logger.info(f"\t{data.calc_path}")

    for data in models:
        print(f"Predicting for model: {data.calc_name}")
        logger.info(f"Predicting for model: {data.calc_name}")

        calc = get_calc(model_name, data)

        run_info = {
            "db_path": db_path,
            "calc": calc,  
            "calc_name": data.calc_name,
            "save_dir": Path(f"output/{data.calc_name}"),
            "dft_key": dft_key, 
            "db_name": db_name,
            "is_dft_energy_per_atom": is_dft_energy_per_atom,
            }

        predict_for_db(run_info)
        
        print(f"------------------------------------------------------------------")
        print(f"------------------------------------------------------------------")


def handle_mp(run_info):
    logger.info(f"\t{run_info['calc_name']}")
    calc = get_calc(run_info['model_name'], run_info['data'])
    run_info["calc"] = calc
    predict_for_db(run_info)


def test_mlip_parallel(args):
    """
    This uses the calculator to predict the eneriges of db. Loads the models and
    iterates through them.
    in addition this uses multiprocessing to run in parallel
    #NOTE: 
    for some reason this only works for CHGNet, for others it is not efficient.
    """
    from multiprocessing import Pool
    from multiprocessing import cpu_count

    model_name: str = args.model_type.lower()
    logger = setup_logger(log_prefix=f"{args.db_name}_{model_name}_predict_energy", logger_name="predict_energy")

    logger.info(f"Running for model {model_name}")

    db_path = Path(args.db_path)
    db_name = args.db_name
    dft_key = args.dft_energy_key
    is_dft_energy_per_atom = args.is_dft_energy_per_atom 


    print("Loading the models to eval...")
    logger.info("Loading the models to eval...")


    yaml_file = Path(f"exp_info_{model_name}.yaml") if args.yaml_file is None else Path(args.yaml_file)
    if not yaml_file.exists():
        print(f"The given yaml file not found. Given yaml file: {yaml_file}.")
        raise FileNotFoundError(f"The given yaml file not found. Given yaml file: {yaml_file}.")


    models: list[Exp_info] = load_info(yaml_file)

    print(f"Got {len(models)} models to run")

    logger.info(f"Evaluating the folowing {len(models)} models")
    for data in models:
        logger.info(f"\t{data.calc_name}")
        logger.info(f"\t{data.calc_path}")

    run_infos = []

    logger.info("Making run_info")
    for data in models:

        run_info = {
                "model_name": model_name,
                "data": data,
                "db_path": db_path,
                "calc_name": data.calc_name,
                "save_dir": Path(f"output/{data.calc_name}"),
                "dft_key": dft_key, 
                "db_name": db_name,
                "is_dft_energy_per_atom": is_dft_energy_per_atom,
            }

        run_infos.append(run_info)

    # get number of cores in system
    num_models = len(run_infos)  
    num_cores = 4 if num_models > 4 else num_models
    logger.info(f"Running using multiprocessing  with {num_cores} cores")
    print(f"Running using multiprocessing  with {num_cores} cores")
    
    process = Pool(processes=num_cores)  # Adjust the number of processes as needed
    process.map(handle_mp, run_infos)
        
    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")


def log_run_info(run_info: dict):
    """
    To log all info in run_info dictionary
    """ 
    logger.info(f"Run information:")
    for key, value in run_info.items():
        if key != "calc":
            logger.info(f"\t{key}: \t{value}")


def get_rmse_for_run(file: Path, dft_key, mlip_key):
    """
    Determines the RMSE between dft and predicted energy
    for given pickle file.
    """
    logger.info(f"Loading file: {file}")
    with open(file, 'rb') as f:
        data = pickle.load(f)

    dft_energies = data[dft_key]
    mlip_energies = data[mlip_key]

    if len(dft_energies) != len(mlip_energies):
        raise ValueError(f"Length mismatch: {len(dft_energies)} != {len(mlip_energies)}")

    # calculate RMSE
    sq_error = [(dft - mlip)**2 for dft, mlip in zip(dft_energies, mlip_energies)]

    logger.info(f"{sum(sq_error) / len(sq_error)}")
    return (sum(sq_error) / len(sq_error))**0.5


def get_all_rmse_from_dir(args):
    """
    This extracts the rmse for the passed calcs and then outputs it as a list
    TODO:
        - Add the ability to extract rmse from the saved energies files
    """
    # load the pickle files
    parent_dir = Path(args.eval_dir)
    dft_key = args.dft_rmse_key
    mlip_key = args.mlip_rmse_key
    mptrj_file_name = args.mptrj_file_name
    fe_file_name = args.fe_file_name

    print(f"Running in the dir: {parent_dir}")
    print(f"dft_key: {dft_key}")
    print(f"mlip_key: {mlip_key}")

    rmse_fe = []
    rmse_mptrj = []
    calc_name = []
    for calc_dir in parent_dir.iterdir():
        if calc_dir.is_dir():
            logger.info(f"Processing directory: {calc_dir}")
            # find the files in the directory
            mptrj_file = calc_dir / mptrj_file_name
            fe_file = calc_dir / fe_file_name

            calc_name.append(calc_dir.name)

            if mptrj_file.exists():
                rmse = get_rmse_for_run(mptrj_file, dft_key, mlip_key)
                logger.info(f"RMSE for {mptrj_file}: {rmse}")
                rmse_mptrj.append(rmse)
            else:
                logger.info(f"File {mptrj_file} does not exist in {calc_dir}")
                rmse_mptrj.append(None)

            if fe_file.exists():
                rmse = get_rmse_for_run(fe_file, dft_key, mlip_key)
                logger.info(f"RMSE for {mptrj_file}: {rmse}")
                rmse_fe.append(rmse)
            else:
                logger.info(f"File {mptrj_file} does not exist in {calc_dir}")
                rmse_fe.append(None)

    logger.info("Done")
    logger.info("-----------------------------------------------------")
    logger.info("-----------------------------------------------------")
    logger.info(pd.DataFrame({"calc_name": calc_name,
                        "rmse_mptrj": rmse_mptrj,
                        "rmse_fe": rmse_fe}).to_string())
    print("-----------------------------------------------------")
    print("-----------------------------------------------------")
    data = pd.DataFrame({"calc_name": calc_name,
                        "rmse_mptrj": rmse_mptrj,
                        "rmse_fe": rmse_fe})
    data.sort_values(by="calc_name", inplace=True)
    print(data.to_string())


def eval_db(args):
    if not args.model_type or not args.db_path or not args.db_name:
        print("Please provide all the required arguments: --model_type, --db_path, --db_name")
        sys.exit(1)

    if args.use_multiprocessing: 
        test_mlip_parallel(args)
    else:
        test_mlip(args)


def main():

    parser = argparse.ArgumentParser("Select Model")
    parser.add_argument("--model_type", help="Name of the model ('chgnet', 'mace' etc)", required=False)
    parser.add_argument("--db_path", help="Path to the database", required=False)
    parser.add_argument("--db_name", help="Name of the database", required=False)
    parser.add_argument("--dft_energy_key", help="key to extract energy from Atoms.info[]. If not provided takes it as uncorrected_total_energy from MPTRJ", required=False, default='uncorrected_total_energy')
    parser.add_argument("--is_dft_energy_per_atom", help="If the dft energy for dft_energy_key is per atom then set this to True", required=False, default=False, type=bool)
    parser.add_argument("--get_rmse", help="Set to True to get all Rmse values", required=False, default=False, type=bool)
    parser.add_argument("--eval_dir", help="Provide if --get_rmse is set to True. by default it takes current dir", required=False, default="./", )
    parser.add_argument("--mlip_rmse_key", help="Set the mlip energy key when getting rmse.", required=False, default='mlip_energy_per_atom')
    parser.add_argument("--dft_rmse_key", help="Set the dft energy key when getting rmse.", required=False, default='dft_energy_per_atom')
    parser.add_argument("--mptrj_file_name", help="Set the file name used to save the fe_energy files", required=False, default='mptrj_mlip_dft_energy.pkl')
    parser.add_argument("--fe_file_name", help="Set the file name used to save the fe_energy files", required=False, default='fe_mlip_dft_energy.pkl')

    args = parser.parse_args()

    if args.get_rmse:
        get_all_rmse_from_dir(args)
        sys.exit(1)
    else:
        # make sure all main args are provided
        if not args.model_type or not args.db_path or not args.db_name:
            print("Please provide all the required arguments: --model_type, --db_path, --db_name")
            sys.exit(1)


    test_mlip_parallel(args)


if __name__ == "__main__":
    main()
