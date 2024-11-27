All the following models have been trained with the MSE Loss function, Adam opt. and 1e-4 lr;

AE_cartesian_hd128x64_ld32_att_med_ref (older model, prev. training script)
Hypothesis aggregation through np.median [after] feeding them to the AE (reconstruction aggregation
Best MPJPE -> 42.6369

AE_cartesian_hd128x64_ld32_test_weight_ref
[train_ae_cartesian_weight.py]
Temp = 0.2735 | Weight = 0.1 (weight for the recons. of the single hypotheses, ref. to the script)
Hypothesis aggregation through np.median [before] feeding them to the AE
Best MPJPE -> 42.4599

AE_cartesian_hd128x64_ld32_test_noweight_ref
[train_ae_cartesian_weight.py]
Temp = 0.2735 | Weight = 0.0 (not used)
Hypothesis aggregation through np.median [before] feeding them to the AE
Aggregation of the reconstructions returned the same MPJPE
Best MPJPE -> 42.4153