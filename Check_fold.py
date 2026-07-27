import json

with open("nnUNet_preprocessed\\Dataset001_BraTSAdult\\splits_final.json") as f:
    splits = json.load(f)

val_cases = splits[0]["val"]

print(len(val_cases))
print(val_cases[:10])