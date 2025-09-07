# Symptom-to-disease-and-Medicines-prediction-model-using-XG-BOOST
The model first takes the symptoms from the user and then predicts the top 5 diseases based on XG BOOST model. Those top 5 disease also contains the probability along with the name. Then we predict the medicines using the medical data. we have given a threshold of 60 to match the medicines. We also Provided the Images,side effects of the Medicine.


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

## 👨‍💻 Author

Built by **Dhanush** 🚀


