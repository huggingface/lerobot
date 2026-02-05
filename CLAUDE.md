# X le Robot - Configuration et Progression

## Contexte
"X le robot" - Robot avec deux bras (leader et follower) utilisant LeRobot (Hugging Face) pour la téléopération et la collecte de données.

---

## Statut actuel - 2026-02-05 (SESSION 3)

### Ce qui est fait ✅
1. **Montage du robot** : Assemblage complet ✅
2. **Configuration des bras** : ✅
   - Bras leader : calibré (COM8)
   - Bras follower : calibré (COM7) avec ID "zarax"
   - Calibration sauvegardée : `~/.cache/huggingface/lerobot/calibration/robots/so_follower/zarax.json`
3. **Synchronisation** : Les deux bras fonctionnent en mode miroir ✅
4. **Caméras** : ✅
   - Configurées : USB indice 1 (640x480, 30 FPS)
   - Testées et fonctionnelles avec OpenCV
5. **Téléopération complète** : ✅
   - Bras leader et follower synchronisés
   - Flux vidéo caméra affichés en temps réel
   - Visualisation Rerun activée (`display_data: true`)
   - Pas de déconnexions lors du test
6. **Collecte de données** : ✅
   - Dataset enregistré : `Zarax/zarax-demo` (9 épisodes, 3,491 frames)
   - Fichier de config : `config/record/zarax_record_config_camdroite.yaml`
7. **Entraînement du modèle** : ✅
   - Modèle ACT entraîné sur 20,000 steps
   - Loss finale : 0.035
   - Modèle uploadé : `Zarax/act-zarax-v1`
   - Checkpoint local : `outputs/train/act_zarax_v1/checkpoints/020000/`
8. **Déploiement** : ✅
   - Robot fonctionne en mode autonome avec le modèle entraîné
   - Script simple : `run_model.bat`
   - Config : `config/eval/zarax_eval_simple.yaml`

### Environnement
- Windows 10/11
- Python 3.10 (conda: `lerobot`)
- LeRobot 0.4.4
- Caméras USB : indices 2 et 3 (640x480, 30 FPS)
- Robot ID : "zarax"

---

## 📊 État de la progression

| Étape | Statut | Description |
|-------|--------|-------------|
| 1. Montage | ✅ Complété | Robot assemblé et opérationnel |
| 2. Calibration | ✅ Complété | Bras calibrés (zarax.json) |
| 3. Téléopération | ✅ Complété | Leader/Follower synchronisés avec caméras |
| 4. Collecte de données | ✅ Complété | 9 épisodes enregistrés (Zarax/zarax-demo) |
| 5. Entraînement | ✅ Complété | Modèle ACT entraîné (Zarax/act-zarax-v1) |
| 6. Déploiement | ✅ Complété | Robot fonctionne en mode autonome |

---

## Prochaines étapes - Collecte de données

### 1. Téléopération avec caméras (lerobot-teleoperate) ✅ COMPLÉTÉ
**Status :** Fonctionnel et testé avec succès

**Fichier de configuration :**
- Localisation : `C:\XLeRobot\lerobot\zarax_teleop_config.yaml`
- Paramètres activés : `display_data: true` pour afficher les vidéos

**Résultats du test :**
- ✅ Bras leader et follower synchronisés
- ✅ 2 flux vidéo affichés en temps réel
- ✅ Visualisation Rerun active
- ✅ Pas de déconnexions
- ✅ Boucle de téléopération stable (32 Hz)

### 2. Collecte de données (lerobot-record) - ⏭️ PROCHAINE ÉTAPE
**Status :** À faire

Enregistrer des démonstrations de mouvement du robot pour l'apprentissage par imitation.

**Prérequis :**
- Compte Hugging Face (https://huggingface.co/join)
- Token HF pour authentification

**Commande :**
```bash
lerobot-record --config_path C:\XLeRobot\lerobot\zarax_teleop_config.yaml --repo-id <HF_USERNAME>/zarax-demo --num-episodes 5
```

**Ce que ça fait :**
- Ouvre la fenêtre de téléopération
- Enregistre 5 épisodes de démonstration
- Capture les images des 2 caméras
- Crée un dataset Hugging Face

### 3. Entraînement du modèle (lerobot-train) ✅ COMPLÉTÉ
**Status :** Complété avec succès

**Dataset utilisé :** `Zarax/zarax-demo` (9 épisodes)
**Modèle :** ACT (Action Chunking with Transformers)
**Configuration :** `config/training/zarax_train_config_act.yaml`
**Résultats :**
- 20,000 training steps
- Loss finale : 0.035
- Modèle uploadé sur HuggingFace : `Zarax/act-zarax-v1`

### 4. Déploiement et test du modèle ✅ COMPLÉTÉ
**Status :** Solution finale implémentée

**LA SOLUTION SIMPLE : Script run_model.bat**

Pour faire tourner le robot avec le modèle entraîné, utilise simplement :
```bash
.\run_model.bat
```

**Ce que fait le script :**
- ✅ Nettoie automatiquement le dataset de test précédent
- ✅ Lance le robot avec le modèle entraîné
- ✅ N'upload JAMAIS sur HuggingFace
- ✅ Toujours la même commande, fonctionne à chaque fois

**Fichiers impliqués :**
- Script : `run_model.bat`
- Configuration : `config/eval/zarax_eval_simple.yaml`

**Important découvert :**
- LeRobot n'a pas de mode "inference-only" natif
- `num_episodes: 0` termine immédiatement sans faire tourner le robot
- Il FAUT `num_episodes >= 1` pour que le robot tourne
- La solution : script wrapper qui gère le nettoyage automatique

---

## Configuration du robot

### Fichier de configuration YAML
**Localisation :** `C:\XLeRobot\lerobot\zarax_teleop_config.yaml`

**Contient :**
- Configuration du robot follower (SO101, COM7, ID=zarax)
- Configuration du robot leader (SO101, COM8, ID=zarax)
- Caméras OpenCV (indices 2, 3 @ 640x480, 30 FPS)
- Calibration automatiquement chargée depuis zarax.json

**Structure YAML actuelle :**
```yaml
display_data: true

robot:
  type: so101_follower
  port: COM7
  id: zarax
  cameras:
    camera_0:
      type: opencv
      index_or_path: 2
      fps: 30
      width: 640
      height: 480
    camera_1:
      type: opencv
      index_or_path: 3
      fps: 30
      width: 640
      height: 480

teleop:
  type: so101_leader
  port: COM8
  id: zarax
```

**Calibration :**
- Sauvegardée automatiquement lors de `lerobot-calibrate --robot.id=zarax`
- Chemin : `~/.cache/huggingface/lerobot/calibration/robots/so_follower/zarax.json`
- Chargée automatiquement au démarrage du robot

---

## Commandes utiles

### Environnement
```bash
# Activer conda
conda activate lerobot

# Aller au repo
cd C:\XLeRobot\lerobot
```

### Diagnostic
```bash
# Vérifier les caméras
lerobot-find-cameras opencv

# Test rapide OpenCV
python -c "import cv2; cap = cv2.VideoCapture(2, cv2.CAP_DSHOW); print('Camera 2:', cap.isOpened()); cap.release()"
```

### LeRobot commands

#### Téléopération avec caméras (Recommandé)
```bash
# Utiliser le fichier de configuration YAML
lerobot-teleoperate --config_path C:\XLeRobot\lerobot\zarax_teleop_config.yaml
```

#### Calibration du robot
```bash
# Calibrer les bras (si nécessaire)
lerobot-calibrate --robot.type=so101_follower --robot.port=COM7 --robot.id=zarax
```

#### Collecte de données
```bash
# Enregistrer des démonstrations
lerobot-record --config_path C:\XLeRobot\lerobot\zarax_teleop_config.yaml --repo-id <HF_USERNAME>/zarax-demo
```

#### Entraînement
```bash
# Entraîner un modèle
lerobot-train --help
```

---

## Notes techniques

⚠️ **Backend OpenCV** : Windows utilise DirectShow (CAP_DSHOW) - configuré dans `src\lerobot\cameras\utils.py`

⚠️ **Configuration** : Utiliser un fichier YAML pour la configuration complète avec caméras (plus flexible que CLI)

⚠️ **Calibration** : Sauvegardée automatiquement dans zarax.json lors du premier démarrage ou après `lerobot-calibrate`

⚠️ **Caméras** : OpenCV camera config accepte `index_or_path` (entier ou chemin vers fichier vidéo)

⚠️ **Format CLI** : Utiliser `--config_path` (underscore) et non `--config-path` (tiret)

---

## Ressources

- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [Tutorial complet](https://huggingface.co/docs/lerobot/tutorials)
- [Teleoperation guide](https://huggingface.co/docs/lerobot/teleop)
- [Dataset guide](https://huggingface.co/docs/lerobot/datasets)