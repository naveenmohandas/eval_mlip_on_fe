"""
This script will be used to evaluate all models.
"""
import argparse


from utils.general import setup_logger
from utils.predict_energies_db import eval_db, get_all_rmse_from_dir
from utils.get_calculators import test_chgnet_load_info, test_mace_load_info, test_eqv2, test_sevenn, test_grace_load_info


def eval_mlip_on_fe(args):
    model_type = args.model_type
    logger = setup_logger(log_prefix=f"master_eval_{model_type}", logger_name="master_eval")
    logger.info(f"Model type: {model_type}" )

    kwargs = {}
    kwargs['eval_fe'] = args.eval_fe_prop_false
    kwargs['eval_imp'] = args.eval_fe_imp_false

    if not args.eval_fe_prop_false and not args.eval_fe_imp_false:
        logger.info("NOTHING TO EVALUATE AS BOTH eval_fe_prop AND eval_fe_imp ARE SET TO FALSE")
        print("NOTHING TO EVALUATE AS BOTH eval_fe_prop AND eval_fe_imp ARE SET TO FALSE")
        return
    else:
        logger.info(f"Evaluating Fe properties: {args.eval_fe_prop_false}, Impurity interactions: {args.eval_fe_imp_false}")
        print(f"Evaluating Fe properties: {args.eval_fe_prop_false}, Impurity interactions: {args.eval_fe_imp_false}")

    if model_type == "chgnet":
        # logger.info("Evaluating CHGNet")
        test_chgnet_load_info(args, **kwargs)

    elif model_type == "mace":
        # logger.info("Evaluating mace")
        test_mace_load_info(args, **kwargs)

    elif model_type == "sevenn":
        # logger.info("Evaluating mace")
        test_sevenn(args, **kwargs)

    elif model_type =="eqv2":
        # logger.info("eqv2")
        test_eqv2()
    elif model_type == "grace":
        test_grace_load_info(args, **kwargs)

    else:
        print("function not implemented for calculator")




def main():

    # get command line arguements
    parser = argparse.ArgumentParser("Select Model")
    parser.add_argument("--model_type", help="Name of the parent model ('chgnet', 'mace' etc)", required=False, default=None)
    parser.add_argument("--yaml_file", help="yaml file to load for the models", required=False, default=None)
    parser.add_argument("--eval_mlip_on_fe", help="Whether to run calculate the properties of Fe or DB", required=False,  action='store_true')
    parser.add_argument("--eval_db", help="When set to True the calculator is used to predict the energies of list of structures in pickle file", required=False, action='store_true')
    parser.add_argument("--eval_fe_prop_false", help="When set to False, Fe bulk properties are not evaluated", required=False, default=True, action='store_false')
    parser.add_argument("--eval_fe_imp_false", help="When set to False, impurity interactions are not evaluated", required=False, default=True, action='store_false')
    parser.add_argument("--only_plot", help="Set if only want to make plots based on the yaml_file", required=False, action='store_true')

    # all arguement below are for when eval_db is set to True
    parser.add_argument("--db_name", help="Name of the database", required=False)
    parser.add_argument("--db_path", help="Path to the database", required=False)
    parser.add_argument("--dft_energy_key", help="key to extract energy from Atoms.info[]. If not provided takes it as uncorrected_total_energy from MPTRJ", required=False, default='uncorrected_total_energy')
    parser.add_argument("--is_dft_energy_per_atom", help="If the dft energy for dft_energy_key is per atom then set this to True", required=False, default=False, type=bool)
    parser.add_argument("--use_multiprocessing", help="if use the models are submitted using multiprocessing pool.map", action='store_true', required=False)

    # rmse
    parser.add_argument("--eval_dir", help="Provide if --get_rmse is set to True. by default it takes current dir", required=False, default="./", )
    parser.add_argument("--get_rmse", help="Set to get all Rmse values", required=False, action="store_true")
    parser.add_argument("--mlip_rmse_key", help="For when --get_rmse is True. Set the mlip energy key when getting rmse.", required=False, default='mlip_energy_per_atom')
    parser.add_argument("--dft_rmse_key", help="for when --get_rmse is true. Set the dft energy key when getting rmse.", required=False, default='dft_energy_per_atom')
    parser.add_argument("--mptrj_file_name", help="for when --get_rmse is true. Set the file name used to save the fe_energy files", required=False, default='mptrj_mlip_dft_energy.pkl')
    parser.add_argument("--fe_file_name", help="for when --get_rmse is true. Set the file name used to save the fe_energy files", required=False, default='fe_mlip_dft_energy.pkl')
    args = parser.parse_args()

    if args.only_plot:
        print(f"Only making plots")
        if args.yaml_file is None:
            raise ValueError("Please provide the yaml file using --yaml_file argument")
        else:
            from utils.make_plots import make_plot_from_yaml
            make_plot_from_yaml(args)

    elif args.eval_mlip_on_fe:
        if args.model_type is None:
            raise ValueError("Please provide the model type using --model_type argument")

        eval_mlip_on_fe(args)

    elif args.eval_db:
        if args.model_type is None:
            raise ValueError("Please provide the model type using --model_type argument")
        eval_db(args)

    elif args.get_rmse:
        get_all_rmse_from_dir(args)

    else:
        print("No valid argument provided. Please provide one of the following arguments: --eval_mlip_on_fe, --eval_db, --get_rmse, --only_plot")
    



if __name__ == "__main__":
    main()
