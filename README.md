# Détection d'Abstracts Biomédicaux Générés par IA (medRxiv 2026)

Développement et validation de pipelines NLP — combinant **Machine Learning classique (TF-IDF, stylométrie, stacking)** et **Deep Learning (GPT-2, fine-tuning de RoBERTa)** — pour distinguer les abstracts biomédicaux rédigés par des humains de ceux générés par des modèles de langage (LLM).

## Contexte & Données

* **Source :** Abstracts de preprints biomédicaux en anglais collectés sur **medRxiv** (mars–septembre 2026), mélangés à des abstracts générés par plusieurs LLM.
* **Volume :** 10 000 abstracts d'entraînement (7 127 humains, 2 873 IA — **28,7 % d'IA**) et 2 898 abstracts de test.
* **Nettoyage & qualité des données :**
  * **Abstracts vides :** 752 dans le jeu d'entraînement (100 % étiquetés IA) et 223 dans le test, traités par une règle dédiée (`vide → IA`).
  * **Doublons :** 860 doublons exacts supprimés afin d'éviter toute fuite de données en validation croisée → **8 388 abstracts uniques non vides** (6 285 humains, 2 103 IA).
  * **Nettoyage léger :** Suppression uniquement des balises HTML et espaces de bordure ; conservation de la casse, de la ponctuation et des mots vides (une étude d'ablation montre que le prétraitement NLP classique avec suppression des stopwords et racinisation fait chuter le F1 de `0.769` à `0.736`).

## Méthodologie

### 1. Approche classique (`01_ml_tfidf_stylometry.ipynb`)
* **Ingénierie de variables :** 32 variables stylométriques (longueur et régularité des phrases, richesse lexicale TTR/hapax, score de Flesch, ponctuation, résultats chiffrés, marqueurs typographiques et lexique typique des LLM).
* **Modèles de base (validation croisée stratifiée 5 plis) :** Naive Bayes (mots), TF-IDF mots (1–2 grammes) + Régression Logistique, TF-IDF caractères (1–4 grammes, sensible à la casse) + Régression Logistique, et `HistGradientBoostingClassifier` sur les variables stylométriques.
* **Stacking (Super-Learner) :** `HistGradientBoostingClassifier` combinant les probabilités hors pli des deux modèles TF-IDF et les 32 variables stylométriques, validé par **validation croisée imbriquée**.

### 2. Modèles de langage neuronaux (`02_deep_learning_roberta_gpt2.ipynb`)
* **Découpage sans fuite :** 6 039 abstracts d'entraînement, 671 de réglage (choix de la meilleure époque et des seuils) et 1 678 de validation indépendante.
* **Approche A (Perplexité GPT-2) :** Extraction de 5 indicateurs de prévisibilité (log-perplexité, écart-type / *burstiness*, surprise maximale, part top-1 et top-10) sans réentraînement, suivie d'une Régression Logistique.
* **Approche B (Fine-tuning `roberta-base`) :** Entraînement sur GPU en demi-précision (`float16`, `max_len=512`, `AdamW`, `lr=2e-5`, 3 époques, meilleure époque = 2) avec perte `CrossEntropyLoss` pondérée pour gérer le déséquilibre des classes.

## Résultats

### Validation croisée 5 plis — Script 1 (8 388 abstracts non vides)

| Modèle | Seuil optimal | F1 (CV) | Précision | Rappel | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stacking (TF-IDF + Stylométrie)** | **0.55** | **0.865** | **0.885** | **0.845** | **0.979** |
| Stylométrie (32 var.) + Boosting | 0.72 | 0.822 | 0.952 | 0.723 | 0.960 |
| TF-IDF caractères (1–4) + LR | 0.47 | 0.813 | 0.814 | 0.812 | 0.956 |
| TF-IDF mots (1–2) + LR | 0.40 | 0.769 | 0.757 | 0.782 | 0.937 |
| Naive Bayes (mots) | 0.05 | 0.417 | 0.699 | 0.297 | 0.810 |

*(Validation croisée imbriquée du Stacking : **F1 = 0.865** ; F1 estimé avec la règle des abstracts vides : **0.901**.)*

### Comparaison finale sur le jeu de validation indépendant (1 678 abstracts non vides)

| Modèle | Seuil | F1 (Val) | Précision | Rappel | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fine-tuning `roberta-base` (Modèle final)** | **0.65** | **0.892** | **0.942** | **0.848** | **0.989** |
| Script 1 (Stacking ML) | 0.40 | 0.843 | 0.843 | 0.843 | 0.977 |
| Perplexité GPT-2 + LR | 0.49 | 0.407 | 0.297 | 0.644 | 0.604 |

*(En incluant les 223 abstracts vides du jeu de test classés sans erreur, le F1 estimé de `roberta-base` sur le jeu de test atteint **~0.92**.)*

## Analyse critique : Robustesse & Bruit d'étiquetage

* **Impact des artefacts de collecte :** L'analyse exploratoire révèle des différences de mise en forme liées au scraping de medRxiv (titres collés comme `MethodsWe...` chez les humains, tirets longs et caractères non-ASCII chez les IA). Après normalisation complète de la mise en forme, le F1 du stacking passe de `0.865` à **`0.661`**, isolant le signal purement stylistique et soulignant l'enjeu de généralisation face à des LLM ou des pipelines de collecte inédits.
* **Bruit d'étiquetage :** 78 abstracts étiquetés « humains » (1,2 %) obtiennent une probabilité IA $\ge 0.90$, reflétant un usage partiel probable d'outils d'aide à la rédaction par certains auteurs.

