import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# === 1. Load Data ===
df = pd.read_csv('/Users/shraawanilattoo/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/NCSU/Fellowship/tennis_atp-master/atp_matches_2020_working.csv')

# === 2. Filter for US Open R128 ===
main_df = df[(df['tourney_name'] == 'Us Open')].copy()
tourney_df = df[(df['tourney_name'] == 'Us Open') & (df['round'] == 'R128')].copy()
print("Matches found:", len(tourney_df))

# === 3. Helper: Get games won/lost from score string ===
def get_games(score_str, is_winner=True):
    if pd.isna(score_str):
        return 0, 0
    sets = score_str.split()
    wg = lg = 0
    for s in sets:
        try:
            w, l = map(int, s.split('-')[:2])
            wg += w
            lg += l
        except:
            continue
    return (wg, lg) if is_winner else (lg, wg)

# === 4. Feature extraction ===
def extract_player_features(row, is_winner):
    p = 'winner_' if is_winner else 'loser_'
    o = 'loser_' if is_winner else 'winner_'
    p1 = 'w_' if is_winner else 'l_'
    o1 = 'l_' if is_winner else 'w_'

    svpt = row.get(p1 + 'svpt', 1)
    first_in = row.get(p1 + '1stIn', 0)
    first_won = row.get(p1 + '1stWon', 0)
    second_won = row.get(p1 + '2ndWon', 0)
    bp_faced = row.get(p1 + 'bpFaced', 0)

    games_won, games_lost = get_games(row['score'], is_winner)

    return {
        'player_name': row.get(p + 'name'),
        #'rank': row.get(p + 'rank', 200),
        'seed': row.get(p + 'seed', 999),
        'opponent_rank': row.get(o + 'rank', 200),
        'relative_rank_diff': row.get(p + 'rank', 200) - row.get(o + 'rank', 200),
        'games_won': games_won,
        'games_lost': games_lost,
        'games_diff': games_won - games_lost,
        'aces': row.get(p1 + 'ace', 0),
        'double_faults': row.get(p1 + 'df', 0),
        '1st_serve_pct': first_in / svpt if svpt else 0,
        '1st_serve_win_pct': first_won / first_in if first_in else 0,
        '2nd_serve_win_pct': second_won / (svpt - first_in) if svpt - first_in else 0,
        'bp_saved_rate': row.get(p1 + 'bpSaved', 0) / bp_faced if bp_faced else 0,
        'df_rate': row.get(p1 + 'df', 0) / svpt if svpt else 0,
        'ace_rate': row.get(p1 + 'ace', 0) / svpt if svpt else 0,
        'pressure_score': bp_faced + row.get(p1 + 'df', 0) - row.get(p1 + 'ace', 0),
        'domination_score': (games_won - games_lost) + row.get(p1 + 'ace', 0) - row.get(p1 + 'df', 0)
    }

# === 5. Build player-level DataFrame ===
player_rows = []
for _, row in tourney_df.iterrows():
    player_rows.append(extract_player_features(row, True))   # Winner
    player_rows.append(extract_player_features(row, False))  # Loser

players_df = pd.DataFrame(player_rows).dropna()

players_df['serve_dominance'] = players_df['1st_serve_win_pct'] + players_df['2nd_serve_win_pct']
players_df['mental_toughness'] = players_df['bp_saved_rate'] - players_df['df_rate']

# === 6. Define features and target ===
features = [
     'seed', 'opponent_rank', 'relative_rank_diff',
    'games_won', 'games_lost', 'games_diff',
    'aces', 'double_faults', '1st_serve_pct', '1st_serve_win_pct', '2nd_serve_win_pct',
    'bp_saved_rate', 'df_rate', 'ace_rate', 'pressure_score', 'domination_score',
    'serve_dominance', 'mental_toughness'
]
target = 'games_won'

X = players_df[features]
y = players_df[target]

# === 7. Scale and Train Model ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=4, random_state=42)
clf.fit(X_scaled, y)

# === 12. Predict expected round (probability-weighted average) ===
players_df['reseed_score'] = clf.predict_proba(X_scaled).dot(clf.classes_)


# === 8. Predict and Reseed ===
players_df = players_df.sort_values(by='reseed_score', ascending=False).reset_index(drop=True)
players_df['new_seed'] = players_df.index + 1

# === 9. Update original DataFrame with new seeds ===
updated_df = main_df.copy()

# Winners
updated_df = updated_df.merge(players_df[['player_name', 'new_seed']],
                               left_on='winner_name', right_on='player_name', how='left')
updated_df['winner_seed'] = updated_df['new_seed'].combine_first(updated_df['winner_seed'])
updated_df = updated_df.drop(columns=['player_name', 'new_seed'])

# Losers
updated_df = updated_df.merge(players_df[['player_name', 'new_seed']],
                               left_on='loser_name', right_on='player_name', how='left')
updated_df['loser_seed'] = updated_df['new_seed'].combine_first(updated_df['loser_seed'])
updated_df = updated_df.drop(columns=['player_name', 'new_seed'])

# === 10. Save Final CSV ===
updated_df.to_csv('reseeded_players_randomforest.csv', index=False)

# === 11. Plot reseed scores (optional) ===
plt.figure(figsize=(10, 5))
plt.plot(players_df['new_seed'], players_df['reseed_score'], marker='o')
plt.title('Reseeding Scores - US Open 2020 (R128)')
plt.xlabel('New Seed')
plt.ylabel('Reseed Score')
plt.grid(True)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()
