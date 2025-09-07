# Symptom-to-disease-and-Medicines-prediction-model-using-XG-BOOST
The model first takes the symptoms from the user and then predicts the top 5 diseases based on XG BOOST model. Those top 5 disease also contains the probability along with the name. Then we predict the medicines using the medical data. we have given a threshold of 60 to match the medicines. We also Provided the Images,side effects of the Medicine.
Got it 👍 You want a **GitHub README.md** that explains your project (dataset → training XGBoost → predicting disease → mapping medicines with images and side effects).

Here’s a good draft you can directly use as your `README.md`:

---

# 🏥 AI-Powered Symptom to Medicine Recommendation System

Remedii is an AI-driven healthcare assistant that predicts diseases from symptoms and recommends the most relevant medicines with their **images** and **side effects**, using a custom medicine dataset.

---

## 🚀 Features

* ✅ Symptom-to-disease prediction using **XGBoost**
* ✅ Top-K disease predictions with confidence probabilities
* ✅ Maps predicted diseases to medicines using **fuzzy matching**
* ✅ Displays **medicine name, image, and side effects**
* ✅ Extensible to new datasets and models

---

## 📂 Dataset

We use the Hugging Face dataset:

```python
from datasets import load_dataset

ds = load_dataset("zenitsu09/medicine-dataset-v1")
df = ds["train"].to_pandas()
```

Dataset contains:

* **Medicine Name**
* **Uses (disease/condition)**
* **Image URL**
* **Side\_effects**

---

## ⚙️ Training the Model

```python
import xgboost as xgb

dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
dval   = xgb.DMatrix(X_val, label=y_val)

params = {
    "objective": "multi:softprob",
    "num_class": n_classes,
    "max_depth": 8,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "eval_metric": "mlogloss"
}

evallist = [(dtrain, "train"), (dval, "eval")]

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=4000,
    evals=evallist,
    early_stopping_rounds=100,
    verbose_eval=50
)
```

---

## 🧠 Prediction Pipeline

```python
symptoms = ["pain in eye", "foreign body sensation in eye"]
top_predictions = predict_disease(symptoms, bst, top_k=5)

for disease, prob in top_predictions:
    print(f"{disease}: {prob:.3f}")
```

---

## 💊 Medicine Mapping with Images + Side Effects

```python
from fuzzywuzzy import process

# Step 1: Create mapping
disease_to_med = (
    df.groupby("Uses")
    .apply(lambda g: [{"name": row["Medicine Name"], 
                       "image": row["Image URL"], 
                       "side_effects": row["Side_effects"]} for _, row in g.iterrows()])
    .to_dict()
)

# Step 2: Get medicines for predicted disease
def get_medicines_for_disease(predicted_disease, disease_to_med, top_n=5):
    match, score = process.extractOne(predicted_disease, disease_to_med.keys())
    if score > 60:
        return match, disease_to_med[match][:top_n]
    else:
        return None, [{"name": "No medicine found", "image": "-", "side_effects": "-"}]

# Step 3: Show results
for disease, prob in top_predictions:
    print(f"\nPredicted Disease: {disease} (prob: {prob:.3f})")
    matched_use, meds = get_medicines_for_disease(disease, disease_to_med, top_n=5)
    print(f"Matched Uses: {matched_use}")
    print("Suggested Medicines:")
    for med in meds:
        print(f"  - {med['name']}")
        print(f"    Image: {med['image']}")
        print(f"    Side Effects: {med['side_effects']}")
```

---

## 📸 Example Output

```
Predicted Disease: Eye Infection (prob: 0.812)
Matched Uses: Eye infection treatment
Suggested Medicines:
  - Ciprofloxacin
    Image: https://example.com/cipro.jpg
    Side Effects: nausea, headache, dizziness
  - Ofloxacin
    Image: https://example.com/oflox.jpg
    Side Effects: upset stomach, diarrhea
```

---

## 🛠️ Tech Stack

* **Python**
* **XGBoost** – for disease prediction
* **FuzzyWuzzy** – for disease-to-medicine mapping
* **Pandas / HuggingFace Datasets** – for data processing

---

## 📌 Next Steps

* Add a **frontend (Streamlit/Flask)** for interactive use
* Display **medicine images inline** (not just URLs)
* Expand dataset with more diseases & treatments
* Add **dosage recommendations**

---

## 👨‍💻 Author

Built by **Dhanush** 🚀


