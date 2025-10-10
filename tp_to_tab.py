import os
from datasets import load_dataset

def recode_text(text: str) -> str:
    """Decode escape sequences in text while preserving Unicode."""
    replacements = {
        "\\n": "\n",
        "\\t": "\t",
        "\\r": "\r",
        '\\"': '"',
        "\\'": "'",
    }
    for escaped, replacement in replacements.items():
        text = text.replace(escaped, replacement)
    text = text.replace("\\\\", "\\")
    return text

if __name__ == "__main__":
    input_csv = "./data/trustpilot/trustpilot_reviews_2005.csv"
    ds = load_dataset('csv', data_files={'train': input_csv})
    train_df = ds['train']
    by_company = {}
    for i, row in enumerate(train_df):
        row = {
            key: recode_text(value) 
            if isinstance(value, str) 
            else value
            for key, value in row.items()
        }
        row['review_id'] = i
        company = row.pop('company')
        description = row.pop('description')
        if company not in by_company:
            by_company[company] = {"description": description, "records": []}
        by_company[company]["records"].append(row)

    for company, data in by_company.items():
        mean_stars = sum([r['stars'] for r in data['records']]) / len(data['records'])
        var_stars = sum([(r['stars'] - mean_stars)**2 for r in data['records']]) / len(data['records'])
        data.update({"mean_stars": mean_stars, "var_stars": var_stars})

    known_companies = [
        "www.amazon.com",
        "hellofresh.co.uk",
        "flixbus.co.uk",
        "www.audible.co.uk",
        "backmarket.co.uk",
        "www.hsbc.co.uk",
    ]

    known_by_company = {
        company: data 
        for company, data in by_company.items() 
        if company in known_companies
    }

    for company, data in known_by_company.items():
        os.makedirs(f"./data/trustpilot/{company}", exist_ok=True)
        with open(f"./data/trustpilot/{company}/description.txt", "w") as f:
            f.write(data["description"])
        with open(f"./data/trustpilot/{company}/stats.txt", "w") as f:
            f.write(f"Mean stars: {data['mean_stars']}\n")
            f.write(f"Variance of stars: {data['var_stars']}\n")
            f.write(f"Number of reviews: {len(data['records'])}\n")
        with open(f"./data/trustpilot/{company}/train.json", "w") as f:
            import json
            json.dump(data["records"], f)