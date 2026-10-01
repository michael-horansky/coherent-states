import json

sys_name = "Li2_ccpVDZ_N_MO=16"
ds_name = "RNCS"

N = 50
N_sub = 50
rs = "rp"
freeze_basis = True

dist_vals = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.05, 1.15, 1.2, 1.3, 1.4, 1.7, 2.0]



repeats = 20



nodes = {"basis_samples" : ["pkl", "json"], "result_energy_states" : ["csv", "pkl"]}




filenames = []
system_log_meta_filenames = []

for c in [1.0] + dist_vals:
    filenames.append(f"outputs/{sys_name}_dist={c}/root_meta.json")
    system_log_meta_filenames.append(f"outputs/{sys_name}_dist={c}/system/log_meta.json")

datasets = []

for i in range(repeats):
    cur_ds_name = f"{ds_name}_i{i}_{N}_{N_sub}"
    if rs != "ai":
        cur_ds_name += f"_{rs}"
    if freeze_basis == False:
        cur_ds_name += "_nf"

    datasets.append(cur_ds_name)



for fn in filenames:

    with open(fn, "r") as f:
        root_meta = json.load(f)

    for ds in datasets:

        root_meta["data_nodes"][ds] = []
        root_meta["data_bulk_types"][ds] = {}
        root_meta["metadata_types"][ds] = {}
        for node, vals in nodes.items():
            root_meta["data_nodes"][ds].append(node)
            root_meta["data_bulk_types"][ds][node] = vals[0]
            root_meta["metadata_types"][ds][node] = vals[1]

    with open(fn, "w") as f:
        json.dump(root_meta, f, indent=2)

for fn in system_log_meta_filenames:

    with open(fn, "r") as f:
        system_log_meta = json.load(f)

    for ds in datasets:
        if ds not in system_log_meta["measured_datasets"]:
            system_log_meta["measured_datasets"].append(ds)

    with open(fn, "w") as f:
        json.dump(system_log_meta, f, indent=2)



