import pandas as pd

set_type = ['train', 'test', 'valid']
set_data = ['random', 'stratified']

for i in set_data:
    for j in set_type:
        fname = f"{i}_{j}.csv"
        print(fname)
        df = pd.read_csv(fname)

        df = df.drop(["event", "time_months"], axis=1)
        df["image_path"] = df["case_id"].apply(lambda x: f"data/NPC_pre/T1/image/{x}.nii.gz")
        df["mask_path"] = df["case_id"].apply(lambda x: f"data/NPC_pre/T1/label/{x}.nii.gz")
        df["label_path"] = df["case_id"].apply(lambda x: f"data/labels/{x}.json")
        df["radiomics"] = df["case_id"].apply(lambda x: f"data/radiomics/{x}.json")

        print(df)
        df.to_csv(fname, index=False)  # add if you want to save