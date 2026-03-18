# import pandas as pd
# from models.preprocessing import clean_text

# def load_dataset():
#     rows = []

#     with open("data/Resume.csv", "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()

#             if not line:
#                 continue

#             if line.startswith('"') and line.endswith('"'):
#                 line = line[1:-1]

#             parts = line.split(",", 4)

#             if len(parts) == 5:
#                 rows.append(parts)

#     df = pd.DataFrame(rows, columns=[
#         "resume_id",
#         "category",
#         "resume_text",
#         "skills_list",
#         "experience_years"
#     ])

#     # Remove header row if present
#     df = df[df["resume_id"] != "resume_id"]


#     # Convert numeric columns properly
#     df["resume_id"] = pd.to_numeric(df["resume_id"], errors="coerce")
#     df["experience_years"] = pd.to_numeric(df["experience_years"], errors="coerce")

#     df = df.dropna(subset=["resume_id"])
#     # Remove rare categories (less than 5 samples)
#     df = df.groupby("category").filter(lambda x: len(x) >= 5)

#     # 🔥 IMPORTANT STEP (Text preprocessing)
#     df["clean_text"] = df["resume_text"].apply(clean_text)

#     return df

import pandas as pd
from models.preprocessing import clean_text

def load_dataset():

    df = pd.read_csv("data/Resumedataset.csv")

    
    
    df = df.rename(columns={
        "Text": "resume_text"
    })

    
    
    df = df[["category", "job_title", "resume_text"]]

    
    df = df.dropna()

  
    df = df.groupby("category").filter(lambda x: len(x) >= 5)

 
    df["clean_text"] = df["resume_text"].apply(clean_text)

    return df