#!/usr/bin/env python3
"""
Script pour surveiller la progression de l'entraînement LeRobot en temps réel.

Usage:
    python config/monitor_training.py
    python config/monitor_training.py --log-path outputs/train/mon_entrainement/train.log
    python config/monitor_training.py --total-steps 50000
"""

import argparse
import re
import time
from pathlib import Path
from datetime import datetime, timedelta


def parse_log_line(line: str) -> dict | None:
    """Parse une ligne de log pour extraire les informations."""
    # Exemple: step:2K smpl:13K ep:33 epch:3.67 loss:1.355 grdn:53.892 lr:1.0e-05 updt_s:0.191 data_s:0.003
    pattern = r"step:(\d+[KM]?)\s+.*loss:([\d.]+)\s+.*updt_s:([\d.]+)"
    match = re.search(pattern, line)

    if match:
        step_str = match.group(1)
        loss = float(match.group(2))
        update_time = float(match.group(3))

        # Convertir les steps (1K = 1000, 1M = 1000000)
        if step_str.endswith('K'):
            step = int(float(step_str[:-1]) * 1000)
        elif step_str.endswith('M'):
            step = int(float(step_str[:-1]) * 1000000)
        else:
            step = int(step_str)

        return {
            'step': step,
            'loss': loss,
            'update_time': update_time
        }
    return None


def format_time(seconds: float) -> str:
    """Formate les secondes en format lisible."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def format_percentage(value: float) -> str:
    """Formate un pourcentage avec barre de progression."""
    percentage = int(value)
    bar_length = 30
    filled = int(bar_length * value / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    return f"[{bar}] {percentage}%"


def find_latest_wandb_log(output_dir: Path) -> Path | None:
    """Trouve le dernier fichier output.log dans le dossier wandb."""
    wandb_dir = output_dir / "wandb"
    if not wandb_dir.exists():
        return None

    # Trouver tous les dossiers run-*
    run_dirs = sorted(wandb_dir.glob("run-*"), key=lambda p: p.stat().st_mtime, reverse=True)

    if run_dirs:
        # Prendre le plus récent
        latest_run = run_dirs[0]
        output_log = latest_run / "files" / "output.log"
        if output_log.exists():
            return output_log

    return None


def get_last_log_entry(log_path: Path) -> dict | None:
    """Lit la dernière entrée du fichier de log."""
    if not log_path.exists():
        return None

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Chercher la dernière ligne avec step info
        for line in reversed(lines):
            info = parse_log_line(line)
            if info:
                return info
    except Exception as e:
        print(f"Erreur lors de la lecture du log: {e}")

    return None


def monitor_training(log_path: Path, total_steps: int = 100000, refresh_rate: float = 2.0):
    """Surveille l'entraînement en temps réel."""
    print("=" * 70)
    print(">>> SURVEILLANCE ENTRAINEMENT LEROBOT")
    print("=" * 70)
    print(f"Fichier log: {log_path}")
    print(f"Steps totaux: {total_steps:,}")
    print(f"Rafraichissement: {refresh_rate}s")
    print("=" * 70)
    print("\nAppuyez sur Ctrl+C pour arreter\n")

    start_time = None
    previous_step = 0
    steps_history = []

    try:
        while True:
            info = get_last_log_entry(log_path)

            if info:
                current_step = info['step']
                loss = info['loss']
                update_time = info['update_time']

                # Initialiser le temps de départ
                if start_time is None:
                    start_time = datetime.now()
                    previous_step = current_step

                # Calculer la progression
                progress = (current_step / total_steps) * 100
                remaining_steps = total_steps - current_step

                # Calculer la vitesse moyenne
                if current_step > previous_step:
                    steps_per_sec = (current_step - previous_step) / refresh_rate
                    steps_history.append(steps_per_sec)

                    # Garder seulement les 10 dernières mesures pour la moyenne
                    if len(steps_history) > 10:
                        steps_history.pop(0)

                    avg_steps_per_sec = sum(steps_history) / len(steps_history)

                    # Temps restant estimé
                    if avg_steps_per_sec > 0:
                        remaining_time = remaining_steps / avg_steps_per_sec
                    else:
                        remaining_time = 0
                else:
                    avg_steps_per_sec = 0
                    remaining_time = 0

                # Temps écoulé
                elapsed_time = (datetime.now() - start_time).total_seconds()

                # Affichage
                print("\033[2J\033[H")  # Clear screen
                print("=" * 70)
                print(">>> SURVEILLANCE ENTRAINEMENT LEROBOT")
                print("=" * 70)

                print(f"\n[PROGRESSION]")
                print(f"   {format_percentage(progress)}")
                print(f"   Step: {current_step:,} / {total_steps:,}")

                print(f"\n[METRIQUES]")
                print(f"   Loss: {loss:.4f}")
                print(f"   Update time: {update_time:.3f}s")

                print(f"\n[TEMPS]")
                print(f"   Ecoule: {format_time(elapsed_time)}")
                if remaining_time > 0:
                    print(f"   Restant: {format_time(remaining_time)} (estime)")
                    eta = datetime.now() + timedelta(seconds=remaining_time)
                    print(f"   Fin estimee: {eta.strftime('%H:%M:%S')}")

                print(f"\n[VITESSE]")
                if avg_steps_per_sec > 0:
                    print(f"   {avg_steps_per_sec:.2f} steps/s")
                    print(f"   ~{60 * avg_steps_per_sec:.0f} steps/min")

                print("\n" + "=" * 70)
                print(f"Derniere mise a jour: {datetime.now().strftime('%H:%M:%S')}")
                print("Appuyez sur Ctrl+C pour arreter")

                previous_step = current_step

            else:
                print("\033[2J\033[H")  # Clear screen
                print("=" * 70)
                print(">>> SURVEILLANCE ENTRAINEMENT LEROBOT")
                print("=" * 70)
                print("\n[En attente du demarrage de l'entrainement...]")
                print(f"Fichier log: {log_path}")
                print(f"\nDerniere tentative: {datetime.now().strftime('%H:%M:%S')}")

            time.sleep(refresh_rate)

    except KeyboardInterrupt:
        print("\n\n[OK] Surveillance arretee par l'utilisateur.")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Surveiller l'entraînement LeRobot en temps réel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="outputs/train/act_zarax_v1/train.log",
        help="Chemin vers le fichier de log",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=100000,
        help="Nombre total de steps d'entraînement",
    )
    parser.add_argument(
        "--refresh-rate",
        type=float,
        default=2.0,
        help="Fréquence de rafraîchissement en secondes",
    )

    args = parser.parse_args()

    log_path = Path(args.log_path)

    # Si le fichier n'existe pas, essayer de trouver le log wandb automatiquement
    if not log_path.exists():
        # Essayer de trouver le dossier output
        if "train.log" in str(log_path):
            output_dir = log_path.parent
            wandb_log = find_latest_wandb_log(output_dir)
            if wandb_log:
                print(f"[INFO] Fichier train.log non trouve, utilisation du log wandb:")
                print(f"       {wandb_log}")
                log_path = wandb_log
            else:
                print(f"[ERREUR] Aucun fichier de log trouve dans {output_dir}")
                print(f"   L'entrainement n'a peut-etre pas encore demarre.")
                print(f"\n[INFO] Commande pour demarrer l'entrainement:")
                print(f"   lerobot-train --config_path .\\config\\training\\zarax_train_config_act.yaml")
                return
        else:
            print(f"[ERREUR] Le fichier {log_path} n'existe pas.")
            return

    monitor_training(log_path, args.total_steps, args.refresh_rate)


if __name__ == "__main__":
    main()
