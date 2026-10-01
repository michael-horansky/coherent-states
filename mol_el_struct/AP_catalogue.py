from utils.class_Access_Point import AccessPoint

########################### ACCESS POINT CATALOGUE ############################

sep_coef_AP = AccessPoint(
        "AP_separation_coefficient_variation.py",
        "For a given diatomic molecule, calculates its electronic ground state energy for different separation coefficients. The basis sample parameter tensor may be frozen to avoid re-sampling for each separation coefficient value.",
        [
            ["mol", str, None, "Molecule label. Must exist as a key in molecules_abstract.mol_catalogue"],
            ["N", int, None, "Size of basis sample (sans ref state) in the calculation. Time complexity is quadratic in N."],
            ["N_sub", int, None, "Size of subsample for each basis state candidate. Time complexity is linear in N_sub."],
            ["method", str, "LE_Zombie_cov_RSOPM_moment_matching", "Keyword specifying the sampling method used (except for deformed c vals when freezing the basis)."],
            ["sr", str, "rp", "Sign randomisation for parameter tensor. \"ai\": as is. \"rs\": random sign. \"rp\": random complex phase."],
            ["sym", str, "default", "Symmetrisation of parameter tensor across spin-subspaces. \"full\": tensors equal. \"phase\": elementwise magnitudes equal. \"none\": no symmetrisation. \"default\": phase for RHF, none for UHF."],
            ["freeze_basis", int, 1, "Whether the parameter tensor is frozen based on the sampling for the default separation coefficient. 1: frozen. 0: not frozen (much higher time complexity)."],
            ["load_analysis", int, 0, "Whether the CI and CISD solution and derived properties are loaded from the disk for each molecule geometry. 0: no loading, the analysis is calculated from scratch. 1: loading."],
            ["load_base_dist", int, 0, "If 0, the calculation is performed for c=1.0 (default). If 1, it is assumed that the calculation was already performed and the sampled basis is loaded from disk."],
            ["c_restrict", int, 0, "Restricts the calculation to a single value of c. 0: no restriction, code runs for all values of c. n > 0: code is restricted to the value of c at index n-1, with n = 1 corresponding to c = 1.0."],
            ["ds_id", str, "RNCS", "Dataset identifier."],
            ["sys_id", str, "sto3g", "System identifier."],
            ["basis", str, "sto-3g", "System identifier."],
            ["N_MO", int, 0, "Number of spatial molecular orbitals per spin subspace. If zero (default), the full set (equal to number of AOs provided) is used."]
        ]
    )

N_MO_AP = AccessPoint(
        "AP_N_MO_scaling.py",
        "For a given diatomic molecule, calculates its electronic ground state energy for different numbers of molecular orbitals. Either a target accuracy or a rule for choosing sample basis sizes must be provided.",
        [
            ["mol", str, None, "Molecule label. Must exist as a key in molecules_abstract.mol_catalogue"],
            ["N", int, None, "Size of basis sample (sans ref state) in the calculation. Time complexity is quadratic in N."],
            ["N_sub", int, None, "Size of subsample for each basis state candidate. Time complexity is linear in N_sub."],
            ["method", str, "LE_Zombie_cov_RSOPM_moment_matching", "Keyword specifying the sampling method used (except for deformed c vals when freezing the basis)."],
            ["sr", str, "rp", "Sign randomisation for parameter tensor. \"ai\": as is. \"rs\": random sign. \"rp\": random complex phase."],
            ["sym", str, "default", "Symmetrisation of parameter tensor across spin-subspaces. \"full\": tensors equal. \"phase\": elementwise magnitudes equal. \"none\": no symmetrisation. \"default\": phase for RHF, none for UHF."],
            ["load_analysis", int, 0, "Whether the CI and CISD solution and derived properties are loaded from the disk for each molecule geometry. 0: no loading, the analysis is calculated from scratch. 1: loading."],
            ["ds_id", str, "RNCS", "Dataset identifier."],
            ["sys_id", str, "sto3g", "System identifier."],
            ["basis", str, "sto-3g", "System identifier."],
            ["N_MO", int, 0, "Number of spatial molecular orbitals per spin subspace. If zero (default), the full set (equal to number of AOs provided) is used."]
        ]
    )
