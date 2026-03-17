# Your custom modules
from models.data_loader import load_dataset
from models.preprocessing import clean_text
from models.tfidf_model import apply_tfidf

# Sklearn utilities
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# ML Models
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
def main():
    


    df = load_dataset()

  
    df["clean_text"] = df["resume_text"].apply(clean_text)

  
    X, vectorizer = apply_tfidf(df["clean_text"])
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

   
    models = {
        "Linear SVM": LinearSVC(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier()
    }

   
    print("\nModel Comparison:\n")

    for name, model in models.items():

        if name == "Random Forest":
            X_train_fit = X_train.toarray()
            X_test_fit = X_test.toarray()
            X_cv = X.toarray()
        else:
            X_train_fit = X_train
            X_test_fit = X_test
            X_cv = X

        model.fit(X_train_fit, y_train)

        y_pred = model.predict(X_test_fit)
        acc = accuracy_score(y_test, y_pred)

        cv_scores = cross_val_score(model, X_cv, y, cv=5)
        cv_mean = cv_scores.mean()

        print(f"{name}")
        print(f"  Accuracy: {round(acc*100,2)} %")
        print(f"  CV: {round(cv_mean*100,2)} %\n")



if __name__ == "__main__":
    main()