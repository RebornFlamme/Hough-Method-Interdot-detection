# autoresearch — Stage 1 (détection d'interdots)

Expérience où le LLM fait sa propre recherche : améliorer **automatiquement** la détection des interdots dans les Charge Stability Diagrams (CSD), sans apprentissage profond imposé — on part d'une pipeline classique Hough et on itère.

Inspiré de [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Contexte (à lire, le repo n'a pas de CLAUDE.md)

- Package Python `csd` géré par **uv** : toujours `uv run python ...` (jamais `python`/`pip` direct), lint `uv run ruff check`.
- Les images CSD font 150×150 px (fenêtre 0.3 V, pas 2 mV), affichées en `origin="lower"` (lignes = y, colonnes = x).
- Les **interdots** à détecter sont des petits « sticks » à ~45° (`theta ≈ π/4` ± 0.1), ~1–2 px de large, ~4–8 px de long, qui apparaissent comme des **creux très sombres** (≈ −7 pour un bruit ≈ 1).
- Pièges : des **lignes de transition** sombres et longues relient les sticks (faux positifs fréquents, autres angles) ; le bruit a une forte composante **horizontale par ligne**.
- Le masque cible = sticks rendus à intensité 1, flou gaussien σ=0.5, seuil 0.5 (masque binaire uint8).
- Dans skimage, `probabilistic_hough_line(theta=...)` prend l'angle de la **normale** à la droite (`theta_stick − π/2`).

## Setup

Pour démarrer une nouvelle série d'expériences, avec l'humain :

1. **Choisir un tag** : proposer un tag basé sur la date du jour (ex. `oct1`). La branche `autoresearch/<tag>` ne doit pas exister — c'est une série neuve.
2. **Créer la branche, toujours depuis `main`** : `git checkout main` puis `git checkout -b autoresearch/<tag>`. Chaque série repart de la **baseline de `main`** (`detect.py` tel qu'il est sur `main`). Ne jamais repartir d'une branche `autoresearch/*` précédente, ni consulter / réutiliser son code, ses résultats ou son `results.tsv`.
3. **Lire les fichiers du périmètre** (le dossier est petit, lis-les en entier) — et **rien d'autre** (voir « Pas de connaissance du simulateur ») :
   - `autoresearch/prepare.py` — constantes figées, génération des données, `evaluate_dice`. **Ne pas modifier.**
   - `autoresearch/detect.py` — le fichier que tu modifies (pipeline de détection).
4. **Choisir la batch size** avec l'humain = nombre d'images de validation (et de réglage) par évaluation. Défaut **128** (~1.5 s par run). Plus petit = plus rapide mais score plus bruité ; plus grand = plus fiable. Cette valeur `N` est **fixe pour toute la série** : tous les runs utilisent le même `--batch-size N`, sinon les scores ne sont pas comparables. Note-la dans le premier commit / la description de la baseline.
5. **Préparer les données** : `uv run python autoresearch/prepare.py --batch-size N` (ne génère que si `data/ar_train` / `data/ar_val` ont moins de `N` images ; ~1 min pour 128).
6. **Initialiser `autoresearch/results.tsv`** avec seulement la ligne d'en-tête. La baseline sera enregistrée au premier run.
7. **Confirmer et lancer** : présente ce setup à l'humain. **Son accord à cette étape vaut approbation du plan pour toute la boucle** (y compris la règle « plan avant chaque outil » du `CLAUDE.md` global) : ensuite, plus aucune question.

## Expérimentation

Chaque expérience se lance avec : `uv run python autoresearch/detect.py --batch-size N`. Elle évalue `predict(image)` sur les `N` premières images de validation (`data/ar_val`) et imprime un résumé.

**Ce que tu PEUX faire :**
- Modifier `autoresearch/detect.py` — c'est le **seul** fichier que tu édites. Tout est permis : prétraitement, paramètres Hough, filtres morphologiques, filtre d'angle, post-traitement, approche complètement différente, recherche de paramètres sur `load_train()` dans le budget de temps, etc.

**Ce que tu NE PEUX PAS faire :**
- Modifier `autoresearch/prepare.py` (lecture seule : données, budget, batch size, métrique) ni le package `csd/`.
- Modifier le bloc d'évaluation de `detect.py` (`parse_batch_size()` → `evaluate_dice(predict, batch_size)` → résumé imprimé) ni changer `--batch-size` en cours de série.
- Lire les masques ou `sticks.jsonl` de `data/ar_val` (ni directement, ni via `load_dataset`). Seul `evaluate_dice` y touche. Pour régler quoi que ce soit, utilise `load_train(batch_size)` (`N` images + masques, seed différente).
- Installer des paquets ou ajouter des dépendances : uniquement ce qui est déjà dans `pyproject.toml` (numpy, scipy, scikit-image, matplotlib).
- Modifier la métrique : `evaluate_dice` (Dice moyen par image) est la vérité terrain.
- **Pas de connaissance du simulateur** (expérience en aveugle) :
  - ne pas lire le package `csd/` (`generator.py`, `config.py`, `dataset.py`, `simulator.py`…) ni aucun code / doc qui décrit la génération des données ;
  - ne pas lire `sticks.jsonl` ni `meta.json`, **même de `data/ar_train`** : le réglage n'utilise que `load_train()` (images + masques) ;
  - ne pas exploiter de paramètres du générateur connus par ailleurs (plages d'intensité, ratios, largeurs, probabilités…) ;
  - les seules connaissances a priori autorisées sont la section *Contexte* ci-dessus et ce que tu mesures toi-même sur les images / masques de `load_train()`.

**Objectif : le `val_dice` le plus haut possible.**

**Budget de temps** : toute recherche/réglage dans `detect.py` doit tenir dans `TIME_BUDGET` = 60 s (constante de `prepare.py`). Un run complet doit rester **< 5 min** ; au-delà, tue-le et compte-le comme un échec.

**Critère de simplicité** : à résultat égal, le plus simple gagne. Un +0.001 de Dice qui ajoute 30 lignes bricolées ne vaut probablement pas le coup. Supprimer du code à score égal ou meilleur : à garder. Gain ~0 mais code nettement plus simple : à garder.

**Premier run** : établit toujours la baseline — lance `detect.py` tel quel.

## Format de sortie

```
---
val_dice:        0.738981
batch_size:      128
tune_seconds:    0.0
total_seconds:   1.4
```

Extraire la métrique : `grep "^val_dice:\|^total_seconds:" autoresearch/run.log`

## Enregistrer les résultats

Après chaque expérience, ajoute une ligne à `autoresearch/results.tsv` (séparé par des **tabulations**, pas des virgules — les virgules cassent les descriptions). En-tête + 5 colonnes :

```
commit	val_dice	seconds	status	description
```

1. hash git court (7 caractères)
2. `val_dice` obtenu (ex. `0.738981`) — `0.000000` pour un crash
3. `total_seconds` arrondi à `.1f` — `0.0` pour un crash
4. statut : `keep`, `discard` ou `crash`
5. courte description de ce que l'expérience a testé

Exemple :

```
commit	val_dice	seconds	status	description
a1b2c3d	0.738981	2.5	keep	baseline
b2c3d4e	0.751200	2.7	keep	angle_tol 0.6 -> 0.5
c3d4e5f	0.730000	2.4	discard	remove gaussian smoothing
d4e5f6g	0.000000	0.0	crash	top-hat with wrong footprint shape
```

**Ne commite jamais `results.tsv`** (il est dans `.gitignore`). Le graphe de progression se trace avec `autoresearch/analysis.ipynb` (→ `autoresearch/progress.png`).

## La boucle d'expériences

La série tourne sur sa branche dédiée (ex. `autoresearch/oct1`).

BOUCLE INFINIE :

1. Regarder l'état git : branche et commit courants.
2. Modifier `autoresearch/detect.py` avec une idée d'expérience.
3. `uv run ruff check autoresearch/detect.py` (corriger s'il y a des erreurs), puis `git commit -am "<description courte>"`.
4. Lancer : `uv run python autoresearch/detect.py --batch-size N > autoresearch/run.log 2>&1` (tout rediriger — pas de `tee`, ne pas inonder ton contexte).
5. Lire : `grep "^val_dice:\|^total_seconds:" autoresearch/run.log`
6. Sortie vide = crash : `tail -n 50 autoresearch/run.log` pour lire la trace et tenter une correction. Après quelques essais infructueux, abandonner l'idée.
7. Enregistrer la ligne dans `results.tsv`.
8. Si `val_dice` est **strictement meilleur** que le meilleur `keep` : on garde le commit, la branche « avance ».
9. Sinon (égal ou pire, ou crash) : `git reset --hard HEAD~1` pour revenir au dernier état gardé.

Tu es un chercheur autonome : si ça marche, on garde ; sinon on jette ; et la branche avance pour qu'on itère dessus. Si tu es coincé, tu peux revenir en arrière, mais très rarement.

**Crashs** : bug bête (typo, import manquant) → corrige et relance. Idée fondamentalement cassée → statut `crash` dans le TSV et on passe à la suite.

**NE JAMAIS S'ARRÊTER** : une fois la boucle lancée (après le setup), ne demande PAS à l'humain s'il faut continuer, ni « est-ce un bon moment pour s'arrêter ? ». L'humain dort peut-être ou n'est pas devant l'ordinateur, et s'attend à ce que tu travailles *indéfiniment* jusqu'à ce qu'il t'arrête. Si tu manques d'idées, réfléchis plus : réanalyse les images / masques de `load_train()` (erreurs par image, faux positifs / faux négatifs), combine des quasi-succès, essaie des approches plus radicales.

### Pistes d'idées (non exhaustif)

- Régler `ANGLE_TOL` / `ANGLE_MIN_LEN` du filtre d'angle ; filtrer aussi par **longueur** (les sticks font ~4–8 px, les lignes de transition sont longues) ou par excentricité.
- Débruitage du **bruit horizontal par ligne** (soustraire la médiane de chaque ligne avant le seuil).
- Top-hat noir / filtre matché orienté à 45° (noyau en forme de stick) au lieu du seuil brut.
- Seuil local (MAD par fenêtre) plutôt que global ; hystérésis (`skimage.filters.apply_hysteresis_threshold`).
- Recherche aléatoire / coordinate descent des `PARAMS` sur `load_train(batch_size)` dans `TIME_BUDGET`.
- Supprimer la transformée de Hough si un filtre orienté + composantes connexes fait mieux (simplification).
- Ajuster l'épaisseur du masque prédit à celle des cibles (≈ 1–2 px, flou σ=0.5 puis seuil 0.5).
