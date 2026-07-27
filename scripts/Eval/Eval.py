import os
import sys
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from metrics import get_LesionWiseResults

pred_dir = os.path.join(ROOT, "test_predictions_baseline")
gt_dir = os.path.join(ROOT, "test_gt")
sm_baseline = os.path.join(ROOT,"scripts\Eval\summary_baseline.csv")
# sm_DA = os.path.join(ROOT,"scripts\Eval\summary_DA.csv")
cases = sorted(os.listdir(pred_dir))

all_results = []

for case in cases:

    pred_file = os.path.join(pred_dir, case)
    gt_file = os.path.join(gt_dir, case)

    if not os.path.exists(gt_file):
        print("GT missing:", case)
        continue

    print("Evaluating:", case)

    df = get_LesionWiseResults(
        pred_file,
        gt_file,
        challenge_name="BraTS-PED"
    )

    df.insert(0, "Case", case)

    all_results.append(df)

final_df = pd.concat(all_results)

output_path = os.path.join(os.path.dirname(__file__), "metrics_baseline.csv")

final_df.to_csv(output_path, index=False)

print("\nSaved results →", output_path)
print("\nSaved results summary →",sm_baseline)