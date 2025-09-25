# About
This repository contains the scripts to evaluate the MLIPs for Fe.


# Usage
- Clone the repo locally.
    `git clone git@github.com:naveenmohandas/eval_mlip_on_fe.git`

- Install required dependecies
    - For analysing results:  `pip install ase pymatgen pytorch pyyaml seaborn`
    - For evaluating mlips:  also install the MLIP `chgnet`, `mace`, `sevennet`

>[!note]
>It is possible that multiple MLIPs cannot be simulteneously installed in the same environement. Use different environements in that case. 
> Make sure the packages `ase`, `pymatgen`, `pyyaml`, `pytorch`, `seaborn` are present in the environemnts.

## Example snippets 

For evaluating models after training, a  yaml file is required with the below template. `calc_name` is the name of the directory in which the output files will be saved in the `output` directory. `calc_path` is the path to the fine-tuned model. For CHGNet and Sevennet, model names can be used in calc_path to load foundational models. It evaluates the models in series. The yaml file name by default is `exp_info_{model_type}.yaml` where model_type is  `chgnet`, `mace` or `sevennet`.

A template for `exp_info_chgnet.yaml` is given below:
```
1:
  calc_name: CHG2
  calc_path: 0.2.0 
2:
  calc_name: CHG3
  calc_path: 0.2.0 
```

> [!NOTE]
> The commands need to be run within the cloned directory. 
### Running run_mlip_evaluation
To predict the properties related to Fe such as Lattice parameter, vacancy formation energy, and so on.

`python run_mlip_evaluation.py --model_type chgnet --eval_mlip_on_fe`

The script loads the model related info from the `exp_info_{model_type}.yaml`

> [!IMPORTANT]
> Check the output and make sure all relaxations converged. This has to be done manually. 

### Fit to DB
#### CHGNET and MPTRJ dataset

`python run_mlip_evaluation.py --eval_db --model_type chgnet --db_path ./input/mptrj_100k.xyz --db_name mptrj --is_dft_energy_per_atom True --dft_energy_key energy_per_atom`

`--model_type` is for selecting the parent model
`--db_path` gives the path to pickle file with list of structures
`--db_name` a name to be used as prefix while saving the resulting pickle file

By default the model saves the files in `output/{model_name}` with the `model_name` from the input yaml file. 


>[!Note]
> The additional flag of `--dft_energy_per_atom` for chgnet is because it uses corrected energy per atom and in the dataset I use it is saved with key energy_per_atom. Further in my dataset the dft energy is saved as energy per atom, so additional tag `--is_dft_energy_per_atom` needs to be set to True. For mace and other mlips, it by default uses `uncorrected_total_energy`. If using some other dataset specify the dft_energy_key. After predicting for all structures it saves the result-dictionary as a pickle file  with keys `dft_energy_per_atom` and `mlip_energy_per_atom`. 


#### MACE and MPTRJ dataset

`python run_mlip_evaluation.py --model_type mace --db_path ./input/mptrj_100k.xyz --db_name mptrj`


#### For other dataset
`python run_mlip_evaluation.py --model_type mace --db_path ./input/nnip-44-60-w-outliers.xyz --db_name fe`


##### Getting the RMSE after predicting the runs.

`python run_mlip_evaluation.py --eval_db --get_rmse --eval_dir output/`

Evaluates all the dirs. There is option to give the file names by default it looks for
`mprtj_mlip_dft_energy.pkl` and `fe_mlip_dft_energy.pkl`.



### Making plots

The `results.ipynb` can be used to visualise the data after simulations are done.


# List of args:

- For running the predictions on fe_properties
    - `--model_type`
    - `--yaml_file`
    - `--eval_mlip_on_fe`
    - `--eval_fe_prop_false`(optional) -> for test only on Fe properties
    - `--eval_fe_imp_false` (optional) -> for test only on Fe with impurity

Example usage:
   `python run_mlip_evaluation --model_type chgnet --yaml_file filepath --eval_mlip_on_fe --eval_fe_imp`

- For predicting the energies of list of structures saved as a pickle file expects dft energy saved in it.
    - `--model_type`
    - `--eval_db`
    - `--db_name`
    - `--db_path`
    - `--dft_energy_key`
    - `--is_dft_energy_per_atom` (to be used if the energy corresponding to the dft_energy_key is eV/atom)
    - `--use_multiprocessing` (only usable for CHGNet, inefficient for other models)

- for getting the RMSE values from the saved mlip_dft_energy.pkl 
    - `--eval_rmse`
    - `--eval_dir`
    - `--mlip_rmse_key`
    - `--dft_rmse_key`
    - `--mprtj_file_name`
    - `--fe_file_name`


# To evaluate calculators that are not implemented


Load the calculator and call the `eval_calculator` function. For example for grace the below script can be used to evaluate the mlip:

```
from tensorpotential.calculator import grace_fm
from utils.main_eval import eval_calculator

model_name = "GRACE-2L-OMAT-medium-ft-AM"
calc = grace_fm(model_name)  
eval_calculator(calc, model_name)
```
> [!IMPORTANT]
> There may be dependency conflicts when using different MLIPs together. These have not been exhaustively tested.
