# Tennis Re-seeding
This repository consist of codes required for resseding the ATP dataset. Follow this link to view the dataset: https://github.com/JeffSackmann/tennis_atp. 
We have used only the singles data from 2015 to 2023.

Collections of codes:
1) ATP Accuracy checker
2) ELO Code for Re-seeding
3) ELO Accuracy checker
4) Jupyter notebook - XGBoost Regressor for Re-seeding the seeds by committee aslo checking the accuracy.
5) Python file - Random Forest for Re-seeding the seeds by committee.

Re-seeding in all the above codes are preformed on the basis of the first round of matches.

The features include:
final_features = [

    'w_2ndWon',
    'l_1stWon',
    'w_1stWon',
    'serve_efficiency',
    'fatigue',
    'match_dominance_score',
    'l_2ndWon'
]


The following graph shows how the original accuracy and the accuracy of XGBoost.


![image](https://github.com/user-attachments/assets/d8424eb3-8a0a-4d4f-b9c3-8866d022b34c)
