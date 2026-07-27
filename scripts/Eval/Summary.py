import pandas as pd
import os

# path tới file metrics
csv_path = os.path.join(os.path.dirname(__file__), "metrics_DA_post.csv")

df = pd.read_csv(csv_path)

print("Total rows:", len(df))
print("Cases:", df["Case"].nunique())
print()

labels = ["WT", "TC", "ET"]

results = []

for label in labels:

    subset = df[df["Labels"] == label]

    dice_mean = subset["LesionWise_Score_Dice"].mean()
    hd95_mean = subset["LesionWise_Score_HD95"].mean()

    results.append({
        "Label": label,
        "Mean_Dice": dice_mean,
        "Mean_HD95": hd95_mean
    })

summary_df = pd.DataFrame(results)

print("===== Fold0 Summary =====")
print(summary_df)

# save
output_path = os.path.join(os.path.dirname(__file__), "summary_DA_post.csv")
summary_df.to_csv(output_path, index=False)

print("\nSaved to:", output_path)