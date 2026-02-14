# === The 20 fine-grained model labels ===
LABELS = [
    'Self-direction: thought',        # 0
    'Self-direction: action',         # 1
    'Stimulation',                    # 2
    'Hedonism',                       # 3
    'Achievement',                    # 4
    'Power: dominance',               # 5
    'Power: resources',               # 6
    'Face',                           # 7
    'Security: personal',             # 8
    'Security: societal',             # 9
    'Tradition',                      # 10
    'Conformity: rules',              # 11
    'Conformity: interpersonal',      # 12
    'Humility',                       # 13
    'Benevolence: caring',            # 14
    'Benevolence: dependability',     # 15
    'Universalism: concern',          # 16
    'Universalism: nature',           # 17
    'Universalism: tolerance',        # 18
    'Universalism: objectivity',      # 19
]

# === COARSE (your annotation) → list of fine-grained indices ===
COARSE_TO_FINE = {
    "SD": [0, 1],
    "ST": [2],
    "HE": [3],
    "AC": [4],
    "PO": [5, 6],
    "FA": [7],
    "SE": [8, 9],
    "TR": [10],
    "CO": [11, 12],
    "HU": [13],
    "BE": [14, 15],
    "UN": [16, 17, 18, 19],
}

# === Fine-grained index → coarse ===
FINE_TO_COARSE = {}
for coarse, fine_list in COARSE_TO_FINE.items():
    for f in fine_list:
        FINE_TO_COARSE[f] = coarse
