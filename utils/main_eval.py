
from pathlib import Path
import argparse
from datetime import date
import sys
import logging

import pandas as pd
from ase.calculators.calculator import Calculator

from utils.eval_fe_bulk import main_vacancy_energy, main_lattice_constant, get_energy_volume, determine_for_calc, get_gb_energy, eval_surface
from utils.eval_fe_substitutional import main_binding_energy
from utils.eval_fe_interstitial import eval_interstitials



try: 
    import torch
    global DEVICE
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except Exception as e:
    print(f"Pytorch is not installed in the environment")



def eval_calculator(calc: Calculator, calc_name: str,**kwargs):


    logger = logging.getLogger("master_eval")
    print("-----------------------------------------------------")
    print(f"Evaluating {calc_name}")
    print("-----------------------------------------------------\n\n")
    logger.info("-----------------------------------------------------")
    logger.info(f"Evaluating {calc_name}")
    logger.info("-----------------------------------------------------\n\n")
    # save all info in a specific directory

    logger.info("creating the save dir")
    exp_dir = Path(f"output/{calc_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"created dir at {exp_dir}")
    
    eval_summary = {'Property': [],
                    'status': []}

    if  kwargs['eval_fe']:
        # lattice constant and vacancy formation energy
        logger.info("\n\n\t\t----calculating lattice constant and vacancy formation energy-----")
        print("----calculating lattice constant and vacancy formation energy-----")
        eval_summary['Property'].append('Lattice parameter')
        eval_summary['Property'].append('vacancy formation energy')

        try:
            lattice_contsant = main_lattice_constant(calc, calc_name)
            logger.info("Calculating the vacancy formation energy")
            lattice_contsant['vac_energy'] = main_vacancy_energy(calc, calc_name)
            pd_lattice = pd.DataFrame(lattice_contsant)
            pd_lattice.to_csv(f"{exp_dir}/lattice_vac_energy.csv")
            del pd_lattice

            eval_summary['status'].append('Completed')
            eval_summary['status'].append('Completed')


        except Exception as e:
            eval_summary['status'].append('Failed')
            eval_summary['status'].append('Failed')

            logger.error(f"Error while evaluating lattice constant and vacancy formation energy: {e}")
            print("Error while evaluating lattice constant and vacancy formation energy. Check log file for error")

        # logger.info("\n\n\n")
        logger.info("\n\n\t\t----calculating elastic tensor-----")
        print("----calculating elastic tensor-----")
        eval_summary['Property'].append('Elastic tensor')
        try:
            pd_elastic = determine_for_calc(calc, calc_name, verbose=False)
            print(pd_elastic.head(10))
            pd_elastic.to_csv(f"{exp_dir}/elastic_tensor.csv")
            del pd_elastic
            eval_summary['status'].append('Completed')

        except Exception as e:
            eval_summary['status'].append('Failed')
            logger.error(f"Error while evaluating elastic tensor: {e}")
            print("Error while evaluating elastic tensor. Check log file for error")

        logger.info("\n\n\t\t----calculating energy volume-----")
        print("----calculating energy volume-----")

        eval_summary['Property'].append('Energy Volume')
        try:
            pd_ev = get_energy_volume(calc)
            pd_ev.to_csv(f"{exp_dir}/energy_volume.csv")
            del pd_ev
            eval_summary['status'].append('Completed')
        except Exception as e:
            logger.error(f"Error while evaluating energy volume: {e}")
            print("Error while evaluating energy volume. Check log file for error")
            eval_summary['status'].append('Failed')
        logger.info("\n\n\n")

        # evaluate surfaces
        eval_summary['Property'].append('Surface energies')
        logger.info("\n\n\t\t----calculating surface energy-----")
        print("----calculating surface energy-----")

        try:
            pd_surfaces = eval_surface(calc=calc, save_dir=exp_dir)
            print(pd_surfaces.head(20))
            pd_surfaces.to_csv(f"{exp_dir}/gb_surfaces.csv")
            eval_summary['status'].append('Completed')

        except Exception as e:
            eval_summary['status'].append('Failed')
            logger.error(f"Error while evaluating surface energies: {e}")
            print("Error while evaluating surface energies. Check log file for error")

        # GB energies
        logger.info("\n\n\t\t----calculating Gb energy-----")
        print("----calculating Gb energy-----")

        eval_summary['Property'].append('Grain boundaries')
        try:
            pd_gb = pd.read_pickle("./input/gb_atoms.pkl")
            pd_gb = get_gb_energy(pd_gb, calc)

            # # remove the column with structures
            print(pd_gb.head(20))
            pd_gb.to_csv(f"{exp_dir}/gb_energies.csv")
            del pd_gb
            eval_summary['status'].append('Completed')
        except Exception as e:
            eval_summary['status'].append('Failed')
            logger.error(f"Error while evaluating grain boundary energies: {e}")
            print("Error while evaluating grain boundary energies. Check log file for error")


    if kwargs['eval_imp']:

        logger.info("\n\n\t\t-----------------------------EVALUATING IMPURITIES------------------------------")
        print("\n\n\t\t-----------------------------EVALUATING IMPURITIES------------------------------")

        # binding energies
        logger.info("\n\n\n")
        logger.info("\n\t\t----calculating binding energy-----")
        print("----calculating binding energy-----")

        imp_x_list = ['Al', 'Cr', 'Cu', 'Nb', 'Ni', 'Mo', 'Si', 'Sn', 'Ti',  'V',  'Zn', 'vac']
        imp_y_list = ['Al', 'Cr', 'Cu', 'Nb', 'Ni', 'Mo', 'Si', 'Sn', 'Ti',  'V',  'Zn', 'vac']

        imp_x_ys = []
        eval_summary['Property'].append('Binding Energies')
        # create list of impurities in format x-y to run
        try:
            for impx in imp_x_list:
                for impy in imp_y_list:
                    imp_x_ys.append(f"{impx}-{impy}")
                imp_y_list.remove(impx)

            pd_binding:pd.DataFrame = main_binding_energy(calc, calc_name, imp_xy_list=imp_x_ys, fmax=0.001,
                                               save_dir=exp_dir)
            pd_binding.to_csv(f"{exp_dir}/binding_energy.csv")
            # print(pd_binding.head(10))
            del pd_binding
            eval_summary['status'].append('Completed')
        except Exception as e:
            logger.error(f"Error while evaluating binding energies: {e}")
            print("Error while evaluating binding energies. Check log file for error")
            eval_summary['status'].append('Failed')
        #

        # logger.info("\n\n\n")
        logger.info("\n\n\t\t----calculating interstitial-----")
        print("----calculating interstitial-----")
        imps = ['C', 'N', 'O']
        e_octs, e_tets, e_subs = [], [], []

        eval_summary['Property'].append('Interstitial Energies')
        try:
            for imp in imps:
                e_oct, e_tet, e_sub =  eval_interstitials(imp, calc, exp_dir)
                e_octs.append(e_oct)
                e_tets.append(e_tet)
                e_subs.append(e_sub)

            dct_int = { 'imps': imps,
                       'e_oct': e_octs,
                        'e_tet': e_tets,
                        'e_sub': e_subs,
                    }
            pd_int = pd.DataFrame(dct_int)
            pd_int.to_csv(f"{exp_dir}/interstitials.csv")
            del pd_int
            eval_summary['status'].append('Completed')
        except Exception as e:
            logger.error(f"Error while evaluating interstitials: {e}")
            print("Error while evaluating interstitials. Check log file for error")
            eval_summary['status'].append('False')

    pd_summary = pd.DataFrame(eval_summary)

    logger.info("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    logger.info(f"Done evaluatingEvaluating {calc_name}")
    logger.info(pd_summary.head(20))
    logger.info("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxn\n\n\n")

    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(f"Done evaluatingEvaluating {calc_name}")
    print(pd_summary.head(20))
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
