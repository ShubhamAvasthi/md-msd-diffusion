# md-msd-diffusion
Mean Squared Distance and Diffusion Coefficient Calculation Script for LAMMPS Molecular Dynamics Output

## Installations
1. To install argparse:
   ```console
   $ pip install --upgrade argparse
   ```
1. To install sklearn:
   ```console
   $ pip install --upgrade sklearn
   ```
1. To install tqdm:
   ```console
   $ pip install --upgrade tqdm
   ```

## Running the script
1. To get help about running the script:
    ```console
    $ python msd-diffusion.py --help
    ```
1. To run the script:
    ```console
    $ python msd-diffusion.py "path_to_data_file" "path_to_dump_file" adsorbent_atom_id_start adsorbent_atom_id_end adsorbate_atom_id_start adsorbate_atom_id_end layer_size --show_graph
    ```
    For example,
    ```console
    $ python msd-diffusion.py 500w_25h202_5pnp_sci_npt_100ps2.data 500w_25h202_5pnp_sci_npt_100ps2.dump 1 149 150 1649 5.50 --show_graph
    ```
    `--show_graph` is optional and shows the MSD vs timesteps graph for quick reference.
