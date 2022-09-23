import os

studies = [
    "confidnet_chest_xray",
    "cross_entropy_chest_xray",
    "deepgamblers_chest_xray",
    "devries_chest_xray",
    "confidnet_dermoscopy",
    "cross_entropy_dermoscopy",
    "deepgamblers_dermoscopy",
    "devries_dermoscopy",
    "confidnet_lidc_idri",
    "cross_entropy_lidc_idri",
    "deepgamblers_lidc_idri",
    "devries_lidc_idri",
    "confidnet_rxrx1",
    "cross_entropy_rxrx1",
    "deepgamblers_rxrx1",
    "devries_rxrx1",
]
for study in studies[13:14]:
    if "rxrx1" in study:
        data = "rxrx1all_data"
    if "chest_xray" in study:
        data = "xray_chestall_data"
    if "dermoscopy" in study:
        data = "dermoscopyall_data"
    if "lidc_idri" in study:
        data = "lidc_idriall_data"

    start, _ = data.split("_data")

    exp_group_name = f"{start}_proto_corruptions"
    accelerator = "None"
    if "confidnet" in study:
        exp_name = "confidnet"
    if "cross_entropy" in study:
        exp_name = "cross_entropy"
    if "deepgamblers" in study:
        exp_name = "deepgamblers"
    if "devries" in study:
        exp_name = "devries"
    num_epochs = 1

    in_class_study = f"\[{start}\]"

    confidence_measures_ls = ["det_mcp", "det_pe", "ext"]
    confidence_measures = f"\[{confidence_measures_ls[0]},{confidence_measures_ls[1]},{confidence_measures_ls[2]}\]"
    dropout = 1
    if dropout == 1:
        confidence_measures_ls = [
            "det_mcp",
            "det_pe",
            "mcd_mcp",
            "mcd_pe",
            "mcd_ee",
        ]
        confidence_measures = f"\[{confidence_measures_ls[0]},{confidence_measures_ls[1]},{confidence_measures_ls[2]},{confidence_measures_ls[3]},{confidence_measures_ls[4]}\]"
        exp_name = "mcd"
    if data == "rxrx1all_data":
        in_class_study_ls = [
            "rxrx1allcorrbrhigh",
            "rxrx1allcorrbrhighhigh",
            "rxrx1allcorrbrlow",
            "rxrx1allcorrbrlowlow",
            "rxrx1allcorrgaunoilow",
            "rxrx1allcorrgaunoilowlow",
            "rxrx1allcorrelastichigh",
            "rxrx1allcorrelastichighhigh",
            "rxrx1allcorrmotblrhigh",
            "rxrx1allcorrmotblrhighhigh",
        ]
        in_class_study = f"\[{in_class_study_ls[0]},{in_class_study_ls[1]},{in_class_study_ls[2]},{in_class_study_ls[3]},{in_class_study_ls[4]},{in_class_study_ls[5]},{in_class_study_ls[6]},{in_class_study_ls[7]},{in_class_study_ls[8]},{in_class_study_ls[9]}\]"
    fd_shifts_command = f"fd_shifts study={study} data={data} exp.group_name={exp_group_name} exp.name={exp_name} eval.query_studies.in_class_study={in_class_study} trainer.accelerator={accelerator} trainer.num_epochs={num_epochs} model.dropout_rate={dropout} eval.confidence_measures.test={confidence_measures}"

    # print(fd_shifts_command)
    os.system(fd_shifts_command)
