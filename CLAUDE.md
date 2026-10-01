# Projet — CSD hackathon (générateur + simulateur DQD 5 grilles)

Package Python `csd` (géré par **uv**) : génération de Charge Stability Diagrams, simulateur, et deux challenges starter sous `starter/`.

## Architecture
- `csd/` — package : `dataset.py` (generate/load dataset), `generator.py` (rendu CSD + labels), `simulator.py`, `challenge.py` (API participant `Experiment`), `config.py` (`GENERATOR`, `CHALLENGE` — hyperparamètres figés).
- `starter/stage1_detection/` — challenge 1 (détection) : `generate_data.py`, `explore_data.py`.
- `starter/stage2_optimization/` — challenge 2 (optimisation) : `explore_simulator.py`, `optimize.py`.
- `tests.ipynb` (racine) — notebook **Stage 1 uniquement** (Stage 2 retiré à la demande de l'utilisateur ; ne pas le réintroduire). `hough_mask(image, sigma, k, k_fill, theta_tol, threshold, line_length, line_gap, dilate)` → masque uint8 ; `dice(pred, target)`. Cellule d'optimisation : recherche aléatoire (300 essais, rng 0) maximisant le **Dice moyen par image** sur `data/hough_tune` (n=64, seed 1, généré seulement s'il manque) → `BEST_PARAMS` = `{sigma: 0.29, k: 2.72, k_fill: 5.01, theta_tol: 0.26, threshold: 5, line_length: 2, line_gap: 2, dilate: 1}` ; Dice tune 0.512→0.716, `nb_demo` (hors réglage) 0.519→0.665 (~1 min 20). Grille 4 colonnes : image | masque Hough | masque cible | comparaison (vert TP, rouge FP, bleu FN, Dice en titre). Métrique choisie par l'utilisateur : **Dice**.

## API clés
- Stage 1 : `generate_dataset(n, out_dir, seed=, overwrite=)` → dossier `images.npy` / `masks.npy` / `sticks.jsonl` / `meta.json` ; `load_dataset(dir)` → dict `{images, masks, sticks, meta}` (arrays memory-mapped).
- Stage 2 : `new_experiment(seed=)` → `Experiment` ; `exp.measure(g1..g5, span_h/v, step_h/v)` (image 2D), `exp.scan_1d(direction, span, step, g1..g5)` (coupe 1D), `exp.reveal()` (optimum caché + budget). `g2`/`g4` = centre de fenêtre ; `g1`/`g3`/`g5` = barrières (drift + contraste).
- Scène par défaut : 150×150 px, fenêtre 0.3 V.

## Gotchas / contraintes
- **`generate_dataset` refuse un dossier de sortie non vide** sans `overwrite=True` (lève `FileExistsError`). Les runs longs sont interruptibles et laissent un dataset **partiel** (`.npy` pré-alloués à pleine taille + `sticks.jsonl` incomplet + **pas de `meta.json`** car il est écrit après la boucle). Avant de relancer : vider le dossier (ou passer `--overwrite`).
- Conventions géométriques : images en `origin="lower"`, sticks à `theta≈π/4` (≈1 px de large, 4–6 px de long), interdots = creux **très négatifs** (≈ −7 vs bruit ~1). Dans skimage Hough, l'angle passé est celui de la **normale** → `theta_stick − π/2`.
- Windows : si le kernel VS Code a `data/nb_demo` memory-mappé, régénérer ce dossier (`overwrite=True`, ex. via `nbconvert --execute`) échoue avec `OSError [Errno 22]` → redémarrer le kernel d'abord. Pour exécuter le notebook en CLI : `--ExecutePreprocessor.kernel_name=c12-hackathon`.
- Quand l'utilisateur demande d'« afficher dans le nb », il veut la **sortie enregistrée dans la cellule**, pas juste le code. Pour l'injecter sans régénérer `nb_demo` : exécuter un sous-ensemble de cellules avec `uv run --with nbclient` (kernel `c12-hackathon`, `resources={'metadata': {'path': '.'}}`), puis copier `outputs` dans la cellule de `tests.ipynb`.
- `data/` est dans `.gitignore` (datasets régénérables) → vider sans risque.
- Lancer les scripts via `uv run python <script> ...` (jamais `python`/`pip` direct). Formatage/lint : `uv run ruff`.
- Un `.ipynb` vide (0 octet) casse le lecteur de notebooks (JSON EOF) → le supprimer et recréer plutôt que l'éditer.

## Notebooks / kernel (VS Code)
- `ipykernel` est dans le groupe **dev**, **épinglé à `==6.29.5`** (`uv add --dev "ipykernel==6.29.5"`). **Ne pas repasser en ipykernel 7.x** : la 7.x écrit un `kernel.json` avec `supported_encryption: ["curve"]` + `kernel_protocol_version: 5.5` que l'extension Jupyter installée (ms-toolsai.jupyter 2025.9.1) **filtre au refresh** → le kernel « apparaît puis disparaît » dans le picker. La 6.x produit un `kernel.json` classique accepté partout.
- Kernel enregistré : **« Python (c12-hackathon uv) »** (`.venv\Scripts\python.exe -m ipykernel install --user --name c12-hackathon`), chemin absolu vers `.venv\Scripts\python.exe` (Python 3.12.11). Testé : démarre + importe `csd` OK.
- `.vscode/settings.json` épingle `python.defaultInterpreterPath` sur le `.venv` et met `jupyter.kernels.filter: []`.
- Sélection : ouvrir **le dossier `challenge`** (pas le parent `Explorations`, sinon le `.venv` n'est pas auto-découvert) → *Developer: Reload Window* → *Select Kernel* → « Python (c12-hackathon uv) ».
- Environnement VS Code à surveiller : **deux versions de `ms-python.python`** (2026.4.0 + 2026.6.0) et l'extension preview **`ms-python.vscode-python-envs`** peuvent provoquer des flickers de découverte ; désinstaller l'ancienne `ms-python.python` si instable.

## Autoresearch (boucle autonome Stage 1, calquée sur karpathy/autoresearch)
- Dossier `autoresearch/` : `program.md` (instructions de l'agent — **point d'entrée**), `prepare.py` (**figé** : `data/ar_train` n=64 seed 1, `data/ar_val` n=128 seed 2, `TIME_BUDGET=60` s, `dice`, `evaluate_dice` = Dice moyen par image sur `ar_val`), `detect.py` (**seul fichier édité par l'agent** : `hough_mask` + `filter_by_angle` + `predict`), `analysis.ipynb` (graphe de progression → `progress.png`, running max car Dice ↑).
- Données : `uv run python autoresearch/prepare.py` (~1 min, ne régénère pas si présent).
- Run : `uv run python autoresearch/detect.py > autoresearch/run.log 2>&1` puis `grep "^val_dice:" autoresearch/run.log` (~3 s).
- Baseline (`main`) : `val_dice = 0.738981` (sans filtre d'angle : 0.725532). Filtre d'angle : composantes connexes dont l'axe principal (regionprops, converti en angle x-y) dévie de > `ANGLE_TOL=0.6` rad de π/4 supprimées si longueur ≥ 3 px.
- Une expérience = un commit sur `autoresearch/<tag>` ; keep si strictement meilleur, sinon `git reset --hard HEAD~1`. `results.tsv`, `run.log`, `progress.png` sont gitignorés.
- Démarrage du prochain agent : « Regarde `autoresearch/program.md` et lançons une nouvelle expérience ! Fais d'abord le setup. » L'accord de l'humain au setup vaut approbation de toute la boucle (règle plan-avant-outil du CLAUDE.md global).
- Gotcha Bash tool : dans un heredoc, `\n` peut être réduit en `\n` → pour générer du code avec des `\n` littéraux, utiliser des chaînes brutes `r'''...'''`.
