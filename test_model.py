from models.data_loader import load_dataset
from models.preprocessing import clean_text
from models.tfidf_model import apply_tfidf
from models.classification import train_classifier
from sklearn.model_selection import cross_val_score


df = load_dataset()


df["category"] = df["category"].str.upper()


category_mapping = {

    
    "PYTHON DEVELOPER": "IT",
    "JAVA DEVELOPER": "IT",
    "FRONTEND DEVELOPER": "IT",
    "BACKEND DEVELOPER": "IT",
    "FULL STACK DEVELOPER": "IT",
    "DEVOPS ENGINEER": "IT",
    "MACHINE LEARNING ENGINEER": "IT",
    "DATA SCIENTIST": "IT",
    "DATA SCIENCE": "IT",
    "CLOUD ENGINEER": "IT",
    "DATABASE": "IT",
    "HADOOP": "IT",
    "INFORMATION-TECHNOLOGY": "IT",

    
    "ACCOUNTANT": "FINANCE",
    "BANKING": "FINANCE",
    "FINANCE": "FINANCE",

    
    "TEACHER": "EDUCATION",
    "ARTS": "EDUCATION",

    
    "SALES": "SALES",
    "BUSINESS-DEVELOPMENT": "SALES",
    "PUBLIC-RELATIONS": "SALES",

    
    "ENGINEERING": "ENGINEERING",
    "MECHANICAL ENGINEER": "ENGINEERING",
    "CIVIL ENGINEER": "ENGINEERING",
    "ELECTRICAL ENGINEERING": "ENGINEERING",

    
    "HEALTHCARE": "HEALTHCARE",
    "FITNESS": "HEALTHCARE",

    
    "HR": "HR"
}

df["category"] = df["category"].replace(category_mapping)


df = df.groupby("category").filter(lambda x: len(x) >= 10)


df["clean_text"] = df["resume_text"].apply(clean_text)


X, vectorizer = apply_tfidf(df["clean_text"])


y = df["category"]


model, accuracy, report = train_classifier(X, y)

print("Model Accuracy:", accuracy * 100)
print(report)


from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=cv)
print(report, "\n")

print("Cross Validation Accuracy:", scores.mean() * 100)