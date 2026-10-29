"""Branch lists for slimming the open-data AnaTuple. Names are exactly the tuple's branch names."""

ID_BRANCHES = ["mc_run", "mc_subrun", "mc_nthEvtInFile", "eventID"]

TRUTH_BRANCHES = [
    "mc_vtx", "mc_targetZ", "mc_targetA", "truth_targetID", "truth_vtx_module", "truth_is_fiducial",
    "mc_nFSPart", "mc_FSPartPDG", "mc_FSPartPx", "mc_FSPartPy", "mc_FSPartPz", "mc_FSPartE",
    "mc_primFSLepton", "mc_incoming", "mc_incomingE", "mc_current", "mc_intType", "mc_Q2", "mc_w",
]

RECO_EVENT_BRANCHES = [
    # Tier 0
    "MasterAnaDev_minos_trk_is_ok", "MasterAnaDev_nuHelicity", "MasterAnaDev_muon_qp",
    "MasterAnaDev_minos_used_range", "MasterAnaDev_minos_used_curvature",
    # muon
    "MasterAnaDev_muon_P", "MasterAnaDev_muon_E", "MasterAnaDev_muon_Px", "MasterAnaDev_muon_Py",
    "MasterAnaDev_muon_Pz", "MasterAnaDev_muon_theta", "muon_phi",
    # vertex
    "MasterAnaDev_vtx", "vtx", "MasterAnaDev_vtx_module",
    # calorimetry
    "MasterAnaDev_recoil_E", "MasterAnaDev_recoil_passivecorrected", "MasterAnaDev_hadron_recoil",
    "recoil_energy_nonmuon_nonvtx100mm", "nonvtx_iso_blobs_energy", "n_nonvtx_iso_blobs",
    # counts
    "n_prongs", "multiplicity", "MasterAnaDev_hadron_number",
    # derived (kept for validation of the decoder)
    "MasterAnaDev_E", "MasterAnaDev_Q2", "MasterAnaDev_W", "MasterAnaDev_leptonE",
]

RECO_PRONG_BRANCHES = [
    "MasterAnaDev_pion_P", "MasterAnaDev_pion_E", "MasterAnaDev_pion_T", "MasterAnaDev_pion_theta",
    "MasterAnaDev_pion_phi", "MasterAnaDev_pion_Px", "MasterAnaDev_pion_Py", "MasterAnaDev_pion_Pz",
    "MasterAnaDev_hadron_isExiting", "MasterAnaDev_hadron_tm_PDGCode", "MasterAnaDev_hadron_tm_fraction",
    "MasterAnaDev_proton_P_fromdEdx", "MasterAnaDev_proton_E_fromdEdx", "MasterAnaDev_proton_T_fromdEdx",
    "MasterAnaDev_proton_Px_fromdEdx", "MasterAnaDev_proton_Py_fromdEdx", "MasterAnaDev_proton_Pz_fromdEdx",
    "MasterAnaDev_proton_theta", "MasterAnaDev_proton_phi", "MasterAnaDev_proton_score1",
    "MasterAnaDev_sec_protons_P_fromdEdx", "MasterAnaDev_sec_protons_E_fromdEdx",
    "MasterAnaDev_sec_protons_T_fromdEdx", "MasterAnaDev_sec_protons_Px_fromdEdx",
    "MasterAnaDev_sec_protons_Py_fromdEdx", "MasterAnaDev_sec_protons_Pz_fromdEdx",
    "MasterAnaDev_sec_protons_theta_fromdEdx", "MasterAnaDev_sec_protons_proton_scores1",
]

RECO_TREE_BRANCHES = ID_BRANCHES + TRUTH_BRANCHES + RECO_EVENT_BRANCHES + RECO_PRONG_BRANCHES
TRUTH_TREE_BRANCHES = ID_BRANCHES + TRUTH_BRANCHES
