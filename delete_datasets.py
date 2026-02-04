#!/usr/bin/env python3
"""
Script pour supprimer les datasets LeRobot locaux.

Usage:
    python delete_datasets.py                    # Mode interactif
    python delete_datasets.py --user Zarax       # Supprimer datasets d'un utilisateur
    python delete_datasets.py --all              # Supprimer tous les datasets
    python delete_datasets.py --all --no-confirm # Supprimer sans confirmation
"""

import argparse
import shutil
from pathlib import Path


def get_lerobot_cache_dir() -> Path:
    """Retourne le répertoire de cache LeRobot."""
    return Path.home() / ".cache" / "huggingface" / "lerobot"


def list_datasets(cache_dir: Path) -> dict[str, list[Path]]:
    """Liste tous les datasets par utilisateur."""
    datasets = {}

    if not cache_dir.exists():
        return datasets

    # Parcourir tous les sous-dossiers sauf 'calibration'
    for user_dir in cache_dir.iterdir():
        if user_dir.is_dir() and user_dir.name != "calibration":
            user_datasets = [d for d in user_dir.iterdir() if d.is_dir()]
            if user_datasets:
                datasets[user_dir.name] = user_datasets

    return datasets


def print_datasets(datasets: dict[str, list[Path]]) -> None:
    """Affiche la liste des datasets."""
    if not datasets:
        print("❌ Aucun dataset trouvé.")
        return

    print("\n📂 Datasets trouvés:")
    for user, user_datasets in datasets.items():
        print(f"\n  👤 {user}:")
        for dataset in user_datasets:
            size = get_dir_size(dataset)
            print(f"    - {dataset.name} ({format_size(size)})")


def get_dir_size(path: Path) -> int:
    """Calcule la taille d'un répertoire."""
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except Exception:
        pass
    return total


def format_size(size: int) -> str:
    """Formate la taille en unités lisibles."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def delete_user_datasets(cache_dir: Path, username: str, confirm: bool = True) -> None:
    """Supprime tous les datasets d'un utilisateur."""
    user_dir = cache_dir / username

    if not user_dir.exists():
        print(f"❌ Aucun dataset trouvé pour l'utilisateur '{username}'")
        return

    datasets = [d for d in user_dir.iterdir() if d.is_dir()]

    if not datasets:
        print(f"❌ Aucun dataset trouvé pour l'utilisateur '{username}'")
        return

    print(f"\n🗑️  Datasets à supprimer pour '{username}':")
    for dataset in datasets:
        size = get_dir_size(dataset)
        print(f"  - {dataset.name} ({format_size(size)})")

    if confirm:
        response = input(f"\n⚠️  Supprimer tous les datasets de '{username}' ? (oui/non): ")
        if response.lower() not in ["oui", "yes", "y", "o"]:
            print("❌ Suppression annulée.")
            return

    try:
        shutil.rmtree(user_dir)
        print(f"✅ Tous les datasets de '{username}' ont été supprimés.")
    except Exception as e:
        print(f"❌ Erreur lors de la suppression: {e}")


def delete_all_datasets(cache_dir: Path, confirm: bool = True) -> None:
    """Supprime tous les datasets de tous les utilisateurs (garde la calibration)."""
    datasets = list_datasets(cache_dir)

    if not datasets:
        print("❌ Aucun dataset à supprimer.")
        return

    print_datasets(datasets)

    total_size = sum(
        get_dir_size(dataset)
        for user_datasets in datasets.values()
        for dataset in user_datasets
    )

    print(f"\n📊 Taille totale: {format_size(total_size)}")

    if confirm:
        response = input("\n⚠️  Supprimer TOUS les datasets (la calibration sera conservée) ? (oui/non): ")
        if response.lower() not in ["oui", "yes", "y", "o"]:
            print("❌ Suppression annulée.")
            return

    deleted_count = 0
    for username in datasets.keys():
        user_dir = cache_dir / username
        try:
            shutil.rmtree(user_dir)
            deleted_count += len(datasets[username])
            print(f"✅ Datasets de '{username}' supprimés.")
        except Exception as e:
            print(f"❌ Erreur lors de la suppression de '{username}': {e}")

    print(f"\n✅ {deleted_count} dataset(s) supprimé(s).")


def interactive_mode(cache_dir: Path) -> None:
    """Mode interactif pour choisir quoi supprimer."""
    datasets = list_datasets(cache_dir)

    if not datasets:
        print("❌ Aucun dataset trouvé.")
        return

    print_datasets(datasets)

    print("\n🔧 Options:")
    print("  1. Supprimer les datasets d'un utilisateur spécifique")
    print("  2. Supprimer TOUS les datasets")
    print("  3. Annuler")

    choice = input("\nChoisissez une option (1-3): ").strip()

    if choice == "1":
        username = input("Nom d'utilisateur: ").strip()
        delete_user_datasets(cache_dir, username)
    elif choice == "2":
        delete_all_datasets(cache_dir)
    else:
        print("❌ Suppression annulée.")


def main():
    parser = argparse.ArgumentParser(
        description="Supprimer les datasets LeRobot locaux",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--user",
        type=str,
        help="Supprimer les datasets d'un utilisateur spécifique",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Supprimer tous les datasets de tous les utilisateurs",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Ne pas demander de confirmation",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lister les datasets sans rien supprimer",
    )

    args = parser.parse_args()

    cache_dir = get_lerobot_cache_dir()

    if not cache_dir.exists():
        print(f"❌ Répertoire de cache non trouvé: {cache_dir}")
        return

    print(f"📁 Cache LeRobot: {cache_dir}")

    # Mode liste uniquement
    if args.list:
        datasets = list_datasets(cache_dir)
        print_datasets(datasets)
        return

    # Mode ligne de commande
    if args.user:
        delete_user_datasets(cache_dir, args.user, confirm=not args.no_confirm)
    elif args.all:
        delete_all_datasets(cache_dir, confirm=not args.no_confirm)
    else:
        # Mode interactif
        interactive_mode(cache_dir)


if __name__ == "__main__":
    main()
