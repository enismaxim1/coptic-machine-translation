import json
import regex as re
import pandas as pd

def parse_data_to_csv(data_path: str):
    data_dict = {"translation": [], "norm_group": [], "norm": [], "func": [], "pos": [], "arabic": [], "meta::translation": [], "meta::title": []}

    with open(data_path) as f:
        data = f.read()
        data = re.sub(r"^\d+\.", '', data, flags=re.MULTILINE)
        chunks = data.split("\n\n\n")
        for chunk in chunks:
            lines = chunk.split("\n")
            row = {}
            for line in lines:
                header, entry = line.strip().split("\t")
                row[header] = entry.strip()
                
            for column in data_dict:
                data_dict[column].append(row.get(column, None))
            
    df = pd.DataFrame(data_dict)
    df.to_csv("data.csv")

    # filter out all rows of df where "translation" which contain "..." or "…"
    df = df[~df["translation"].str.contains("…|\.\.\.")]
    pattern = r'\([^)]*\d+:\d+[^)]*\)'
    df['translation'] = df['translation'].str.replace(pattern, '', regex=True)
    df = df[~df['translation'].str.contains('\(|\)')]
    df.to_csv("data.csv")

# data_path = "annis-export4348269550637738805.txt"
# parse_data_to_csv(data_path)

def to_translation_json(df: pd.DataFrame):
    translations = []
    # loop through rows
    for _, row in df.iterrows():
        eng = row['translation']
        cop = row['norm_group']
        translations.append({"translation": {'eng': eng, 'cop': cop}})
    with open('translation.json', 'w') as f:
        json.dump(translations, f, indent=2)

df = pd.read_csv("data.csv")
to_translation_json(df)