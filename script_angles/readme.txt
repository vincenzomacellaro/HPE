To build the appropriate dataset for this script, you need to run the following scripts (in this order):
    1. script/split_dataset.py
        >> Splits the Human3.6M dataset into train/val/test; this part extracts only the positional data

    2. script/get_avg_pose.py
        >> Computes and saves the average pose from the /ref_data/ref_kpts.dat file, needed for pose alignment.

    3. script/calculate_global_scale_factor.py
        >> Computes the global scale factor to scale the poses appropriately.

    2. script_angles/split_angle_dataset.py
        >> Converts the previous dataset into the augmented one w/ rotational data, in order to further process it;
        >> It makes it compatible with the https://github.com/TemugeB/joint_angles_calculate work.