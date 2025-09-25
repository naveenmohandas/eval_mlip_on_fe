

mace_run_train \
    --name="MACE" \
    --foundation_model="mace-mp-0b3-medium.model" \
    --multiheads_finetuning=True \
    --train_file="../data-set/train_set.xyz" \
    --valid_file="../data-set/val_set.xyz" \
    --test_file="../data-set/test_set.xyz" \
    --energy_weight=1.0 \
    --forces_weight=1.0 \
    --energy_key="uncorrected_total_energy" \
    --forces_key="force" \
    --E0s="foundation" \
    --lr=0.0001 \
    --scaling="rms_forces_scaling" \
    --batch_size=10 \
    --max_num_epochs=50 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --pt_train_file="mp" \
    --default_dtype="float64" \
    --keep_checkpoints \
    --device=cuda \
    --save_cpu \
    --seed=3
