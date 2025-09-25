
from pathlib import Path

import logging
from utils.general import setup_logger, Exp_info, load_info
from utils.predict_energies_db import eval_db, get_all_rmse_from_dir
from utils.main_eval import eval_calculator

try: 
    import torch
    global DEVICE
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except Exception as e:
    print(f"Pytorch is not installed in the environment")

def test_sevenn(args, **kwargs):
    from sevenn.sevennet_calculator import SevenNetCalculator

    yaml_path = Path(args.yaml_file) if args.yaml_file else Path("./exp_info_sevenn.yaml")
    models: list[Exp_info] = load_info(yaml_path)

    for data in models:
        model_name = data.calc_name
        if model_name =='Sevenn':
            model_path = data.calc_path
            calc = SevenNetCalculator(data.calc_path, device=DEVICE)  # 7net-0, SevenNet-0, 7net-0_22May2024, 7net-0_11July2024 ...

        else:
            model_path = Path(data.calc_path)
            calc = SevenNetCalculator(model_path, device=DEVICE)  # 7net-0, SevenNet-0, 7net-0_22May2024, 7net-0_11July2024 ...
        parent_model_dir = Path(data.parent_dir_path) if data.parent_dir_path is not None else None

        eval_calculator(calc,model_name, parent_model_dir=parent_model_dir, **kwargs)


def test_mace_load_info(args, **kwargs):
    from mace.calculators import mace_mp

    yaml_path = Path(args.yaml_file) if args.yaml_file else Path("exp_info_mace.yaml")
    models: list[Exp_info] = load_info(yaml_path)
    logger = logging.getLogger("master_eval")
    logger.info(f"-----------------------------------------------------\n {models = }")
    logger.info("-----------------------------------------------------")
    for data in models:
        model_path = Path(data.calc_path)
        model_name = data.calc_name
        parent_model_dir = Path(data.parent_dir_path) if data.parent_dir_path is not None else None
        calc = mace_mp(model=model_path, device=DEVICE)

        logger.info("Calc info:")
        logger.info("\t\t model_name = %s", model_name)
        logger.info("\t\t model_path = %s", model_path)
        logger.info("\t\t parent_model_dir = %s", parent_model_dir)

        eval_calculator(calc,model_name, parent_model_dir=parent_model_dir, **kwargs)


def test_chgnet_load_info(args, **kwargs):
    from chgnet.model import CHGNetCalculator, CHGNet

    yaml_path = Path(args.yaml_file) if args.yaml_file else Path("exp_info_chgnet.yaml")
    models: list[Exp_info] = load_info(yaml_path)

    logger = logging.getLogger("master_eval")
    logger.info(f"-----------------------------------------------------\n {models = }")
    logger.info("-----------------------------------------------------")
    for data in models:
        model_path = Path(data.calc_path)
        model_name = data.calc_name

        # handle foundational models
        if data.calc_name.lower() == "chg2":
            # logger.info("Running for out of box CHG2")
            model = CHGNet.load(model_name = '0.2.0')

        elif data.calc_name.lower() == "chg3":
            # logger.info("Running for out of box CHG3")
            model = CHGNet.load(model_name = '0.3.0')

        else:
            # custom models from the yaml file
            model_path = Path(data.calc_path)
            model = CHGNet.from_file(model_path)
        parent_model_dir = Path(data.parent_dir_path) if data.parent_dir_path is not None else None
            # logger.info(f"Running for custom model: {model_path}\n parent_model_dir = {parent_model_dir} ")


        logger.info("Calc info:")
        logger.info("\t\t model_name = %s", model_name)
        logger.info("\t\t model_path = %s", model_path)
        logger.info("\t\t parent_model_dir = %s", parent_model_dir)

        calc = CHGNetCalculator(model)
        eval_calculator(calc,model_name, parent_model_dir=parent_model_dir, **kwargs)

