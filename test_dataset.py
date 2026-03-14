from models.data_loader import load_dataset
from models.skill_alignment import (
    generate_skill_templates,
    extract_skills,
    calculate_alignment
)

import pandas as pd

df = load_dataset()

# Generate automatic templates
templates = generate_skill_templates(df, top_n=5)

alignment_results = []

for index, row in df.iterrows():
    category = row["category"].upper()
    resume_text = row["resume_text"]

    if category in templates:
        required_skills = templates[category]

        matched = extract_skills(resume_text, required_skills)
        score = calculate_alignment(matched, required_skills)

        alignment_results.append({
            "category": category,
            "alignment_score": score
        })

alignment_df = pd.DataFrame(alignment_results)

print("\nAverage Alignment (Auto Templates):\n")
print(alignment_df.groupby("category")["alignment_score"].mean())

