### 🎾 Tennis Re-seeding Using Machine Learning

This project explores a **data-driven method for re-seeding players** in ATP tennis tournaments using machine learning. Traditional ATP seeding relies on world rankings, which may not always reflect a player's current form. This project evaluates if **first-round performance** can be used to **dynamically re-seed players** more effectively.

---

### 📁 Contents of the Repository

Collections of codes:
1) ATP Accuracy checker
2) ELO Code for Re-seeding
3) ELO Accuracy checker
4) Jupyter notebook - XGBoost Regressor for Re-seeding the seeds by committee aslo checking the accuracy.
5) Python file - Random Forest for Re-seeding the seeds by committee.


---

### 📊 Problem Statement

Can we build a **machine learning model** that learns from **Round 1 match statistics** and **predicts who should have been seeded higher**?
This could:

* Challenge the limitations of static ATP rankings.
* Improve fairness and competitiveness in early rounds.
* Highlight underrated players based on form.

---

### 🧠 Methodology

1. **Data Preparation**

   * Source: Jeff Sackmann’s ATP data
   * Focus: Matches from Round 1 (R128 or R64)
   * Features: Aces, double faults, serve stats, break points, and more.

2. **Modeling**

   * Algorithms: Logistic Regression, XGBoost, Random Forest.
   * Target: Whether the player with better first-round stats won.
   * Metric: Accuracy, reseeding quality, feature importance.
   * The features include:
final_features = [

    'w_2ndWon',
    'l_1stWon',
    'w_1stWon',
    'serve_efficiency',
    'fatigue',
    'match_dominance_score',
    'l_2ndWon'
]


3. **Reseeding Logic**

   * Use predicted performance to assign a dynamic rank.
   * Compare original seed vs. ML-based reseed.

---

### 📈 Results

* The model showed strong correlation between first-round stats and actual match outcomes.
* Reseeding based on ML predictions sometimes outperformed ATP seeds in predicting later-round success.
The following graph shows how the original accuracy and the accuracy of XGBoost.


![image](https://github.com/user-attachments/assets/d8424eb3-8a0a-4d4f-b9c3-8866d022b34c)
---

### 🛠 Tools & Libraries

* Python (Pandas, Scikit-learn, Matplotlib, Seaborn)
* Jupyter Notebooks
* Git/GitHub

---

### 📌 Future Work

* Extend model to multiple rounds for more robust re-seeding.
* Incorporate player injury, fatigue, or court type as features.
* Use ranking prediction as input to tournament simulation.

---
