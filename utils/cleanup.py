"""
🧹 Utilitaire de nettoyage pour machine SSH
Exécuter avant chaque notebook pour libérer l'espace disque.
"""

import os
import gc
import shutil
import glob
from pathlib import Path
from typing import List, Tuple, Optional


def get_size_str(size_bytes: int) -> str:
    """Convertit bytes en string lisible."""
    if size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.2f} GB"
    elif size_bytes >= 1e6:
        return f"{size_bytes / 1e6:.2f} MB"
    elif size_bytes >= 1e3:
        return f"{size_bytes / 1e3:.2f} KB"
    return f"{size_bytes} B"


def get_dir_size(path: str) -> int:
    """Calcule la taille totale d'un dossier."""
    total = 0
    try:
        for dp, dn, fn in os.walk(path):
            for f in fn:
                fp = os.path.join(dp, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    except:
        pass
    return total


def cleanup_pycache(base_dir: str = ".", verbose: bool = True) -> int:
    """Supprime tous les __pycache__ et .pyc."""
    cleaned = 0
    
    # __pycache__ directories
    for pycache in glob.glob(os.path.join(base_dir, '**/__pycache__'), recursive=True):
        try:
            size = get_dir_size(pycache)
            shutil.rmtree(pycache)
            cleaned += size
            if verbose:
                print(f"  🗑️ {pycache} ({get_size_str(size)})")
        except Exception as e:
            pass
    
    # .pyc files
    for pyc in glob.glob(os.path.join(base_dir, '**/*.pyc'), recursive=True):
        try:
            size = os.path.getsize(pyc)
            os.remove(pyc)
            cleaned += size
        except:
            pass
    
    return cleaned


def cleanup_jupyter_checkpoints(base_dir: str = ".", verbose: bool = True) -> int:
    """Supprime les .ipynb_checkpoints."""
    cleaned = 0
    
    for checkpoint in glob.glob(os.path.join(base_dir, '**/.ipynb_checkpoints'), recursive=True):
        try:
            size = get_dir_size(checkpoint)
            shutil.rmtree(checkpoint)
            cleaned += size
            if verbose:
                print(f"  🗑️ {checkpoint} ({get_size_str(size)})")
        except:
            pass
    
    return cleaned


def cleanup_huggingface_cache(delete: bool = False, verbose: bool = True) -> Tuple[int, bool]:
    """
    Gère le cache HuggingFace (~/.cache/huggingface).
    Par défaut, affiche la taille mais ne supprime pas.
    """
    hf_cache = os.path.expanduser('~/.cache/huggingface')
    
    if not os.path.exists(hf_cache):
        return 0, False
    
    size = get_dir_size(hf_cache)
    
    if verbose:
        print(f"  📦 Cache HuggingFace: {get_size_str(size)}")
        if not delete:
            print(f"     ⚠️ Pour supprimer: cleanup_huggingface_cache(delete=True)")
    
    if delete:
        try:
            shutil.rmtree(hf_cache)
            if verbose:
                print(f"  ✅ Cache HuggingFace supprimé!")
            return size, True
        except Exception as e:
            if verbose:
                print(f"  ❌ Erreur: {e}")
            return 0, False
    
    return size, False


def cleanup_torch_cache(delete: bool = False, verbose: bool = True) -> Tuple[int, bool]:
    """Gère le cache PyTorch (~/.cache/torch)."""
    torch_cache = os.path.expanduser('~/.cache/torch')
    
    if not os.path.exists(torch_cache):
        return 0, False
    
    size = get_dir_size(torch_cache)
    
    if verbose:
        print(f"  🔥 Cache PyTorch: {get_size_str(size)}")
    
    if delete:
        try:
            shutil.rmtree(torch_cache)
            if verbose:
                print(f"  ✅ Cache PyTorch supprimé!")
            return size, True
        except:
            return 0, False
    
    return size, False


def cleanup_old_embeddings(
    embedding_dir: str = "./data/embeddings",
    keep_latest: bool = True,
    verbose: bool = True
) -> int:
    """
    Supprime les anciens fichiers d'embeddings.
    Garde les plus récents si keep_latest=True.
    """
    if not os.path.exists(embedding_dir):
        return 0
    
    cleaned = 0
    
    # Grouper par préfixe (X_train_*, X_kaggle_*)
    prefixes = {}
    for f in os.listdir(embedding_dir):
        if f.endswith('.npy'):
            # Extraire préfixe (e.g., "X_train_text" de "X_train_text_embeddings.npy")
            parts = f.rsplit('_', 1)
            prefix = parts[0] if len(parts) > 1 else f[:-4]
            
            fpath = os.path.join(embedding_dir, f)
            mtime = os.path.getmtime(fpath)
            size = os.path.getsize(fpath)
            
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append((f, fpath, mtime, size))
    
    # Pour chaque groupe, supprimer les anciens
    for prefix, files in prefixes.items():
        if len(files) > 1:
            # Trier par date (plus récent en premier)
            files.sort(key=lambda x: x[2], reverse=True)
            
            # Garder le premier, supprimer les autres
            for f, fpath, mtime, size in files[1:]:
                if not keep_latest:
                    try:
                        os.remove(fpath)
                        cleaned += size
                        if verbose:
                            print(f"  🗑️ {f} ({get_size_str(size)}) - ancien")
                    except:
                        pass
    
    return cleaned


def cleanup_cuda_cache(verbose: bool = True) -> bool:
    """Vide le cache CUDA si disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if verbose:
                # Afficher mémoire GPU
                for i in range(torch.cuda.device_count()):
                    mem_alloc = torch.cuda.memory_allocated(i)
                    mem_reserved = torch.cuda.memory_reserved(i)
                    print(f"  🖥️ GPU {i}: {get_size_str(mem_alloc)} alloué, {get_size_str(mem_reserved)} réservé")
            return True
    except ImportError:
        pass
    except Exception as e:
        if verbose:
            print(f"  ⚠️ CUDA: {e}")
    return False


def cleanup_temp_files(base_dir: str = ".", patterns: List[str] = None, verbose: bool = True) -> int:
    """Supprime les fichiers temporaires."""
    if patterns is None:
        patterns = [
            '*.tmp', '*.temp', '*.bak', '*.swp', '*~',
            '.DS_Store', 'Thumbs.db',
            '*.log'
        ]
    
    cleaned = 0
    
    for pattern in patterns:
        for f in glob.glob(os.path.join(base_dir, '**', pattern), recursive=True):
            try:
                size = os.path.getsize(f)
                os.remove(f)
                cleaned += size
                if verbose:
                    print(f"  🗑️ {f}")
            except:
                pass
    
    return cleaned


def get_disk_usage() -> Tuple[int, int, int]:
    """Retourne (total, used, free) en bytes."""
    return shutil.disk_usage('/')


def garbage_collect(verbose: bool = True) -> int:
    """Force le garbage collection Python."""
    collected = gc.collect()
    if verbose:
        print(f"  🐍 Python GC: {collected} objets collectés")
    return collected


def full_cleanup(
    base_dir: str = ".",
    delete_hf_cache: bool = False,
    delete_torch_cache: bool = False,
    delete_old_embeddings: bool = False,
    verbose: bool = True
) -> dict:
    """
    Nettoyage complet avant exécution d'un notebook.
    
    Args:
        base_dir: Répertoire de base du projet
        delete_hf_cache: Supprimer le cache HuggingFace (ATTENTION: gros!)
        delete_torch_cache: Supprimer le cache PyTorch
        delete_old_embeddings: Supprimer les anciens embeddings
        verbose: Afficher les détails
    
    Returns:
        dict avec les stats de nettoyage
    """
    print("🧹 NETTOYAGE SYSTÈME")
    print("=" * 50)
    
    stats = {
        'pycache': 0,
        'checkpoints': 0,
        'temp': 0,
        'hf_cache': 0,
        'torch_cache': 0,
        'embeddings': 0,
        'cuda_cleared': False,
        'gc_collected': 0
    }
    
    # 1. Garbage collection Python
    print("\n📦 Mémoire Python:")
    stats['gc_collected'] = garbage_collect(verbose)
    
    # 2. Cache CUDA
    print("\n🖥️ CUDA:")
    stats['cuda_cleared'] = cleanup_cuda_cache(verbose)
    
    # 3. Pycache
    print("\n🐍 Python cache:")
    stats['pycache'] = cleanup_pycache(base_dir, verbose)
    if stats['pycache'] > 0:
        print(f"   Total: {get_size_str(stats['pycache'])}")
    
    # 4. Jupyter checkpoints
    print("\n📓 Jupyter checkpoints:")
    stats['checkpoints'] = cleanup_jupyter_checkpoints(base_dir, verbose)
    
    # 5. Fichiers temp
    print("\n📄 Fichiers temporaires:")
    stats['temp'] = cleanup_temp_files(base_dir, verbose=verbose)
    
    # 6. Cache HuggingFace
    print("\n🤗 HuggingFace:")
    hf_size, hf_deleted = cleanup_huggingface_cache(delete=delete_hf_cache, verbose=verbose)
    if hf_deleted:
        stats['hf_cache'] = hf_size
    
    # 7. Cache PyTorch
    print("\n🔥 PyTorch:")
    torch_size, torch_deleted = cleanup_torch_cache(delete=delete_torch_cache, verbose=verbose)
    if torch_deleted:
        stats['torch_cache'] = torch_size
    
    # 8. Anciens embeddings
    if delete_old_embeddings:
        print("\n📊 Anciens embeddings:")
        emb_dir = os.path.join(base_dir, 'data', 'embeddings')
        stats['embeddings'] = cleanup_old_embeddings(emb_dir, keep_latest=True, verbose=verbose)
    
    # Résumé
    print("\n" + "=" * 50)
    total_cleaned = sum([
        stats['pycache'], stats['checkpoints'], stats['temp'],
        stats['hf_cache'], stats['torch_cache'], stats['embeddings']
    ])
    
    print(f"✅ Total nettoyé: {get_size_str(total_cleaned)}")
    
    # Espace disque
    total, used, free = get_disk_usage()
    print(f"💾 Espace disque: {get_size_str(free)} libre / {get_size_str(total)} total ({100*used/total:.1f}% utilisé)")
    
    return stats


def prepare_for_embeddings(
    embedding_dir: str = "./data/embeddings",
    required_space_gb: float = 5.0,
    verbose: bool = True
) -> bool:
    """
    Prépare le système pour calculer de nouveaux embeddings.
    Supprime les anciens si nécessaire pour libérer de l'espace.
    
    Args:
        embedding_dir: Répertoire des embeddings
        required_space_gb: Espace minimum requis en GB
        verbose: Afficher les détails
    
    Returns:
        True si assez d'espace disponible
    """
    print("🔄 PRÉPARATION POUR EMBEDDINGS")
    print("=" * 50)
    
    # Vérifier espace actuel
    total, used, free = get_disk_usage()
    free_gb = free / 1e9
    
    print(f"💾 Espace libre actuel: {free_gb:.2f} GB")
    print(f"📊 Espace requis: {required_space_gb:.2f} GB")
    
    if free_gb >= required_space_gb:
        print("✅ Assez d'espace disponible!")
        return True
    
    print(f"⚠️ Pas assez d'espace! Nettoyage en cours...")
    
    # 1. Supprimer anciens embeddings
    if os.path.exists(embedding_dir):
        print(f"\n🗑️ Suppression des anciens embeddings dans {embedding_dir}...")
        for f in os.listdir(embedding_dir):
            if f.endswith('.npy'):
                fpath = os.path.join(embedding_dir, f)
                size = os.path.getsize(fpath)
                try:
                    os.remove(fpath)
                    if verbose:
                        print(f"   ✅ {f} ({get_size_str(size)})")
                except Exception as e:
                    print(f"   ❌ {f}: {e}")
    
    # 2. Nettoyage général
    full_cleanup(
        base_dir=".",
        delete_hf_cache=False,  # Garder pour éviter re-téléchargement
        delete_torch_cache=True,
        verbose=verbose
    )
    
    # 3. Re-vérifier
    total, used, free = get_disk_usage()
    free_gb = free / 1e9
    
    print(f"\n💾 Espace libre après nettoyage: {free_gb:.2f} GB")
    
    if free_gb >= required_space_gb:
        print("✅ Assez d'espace maintenant!")
        return True
    else:
        print(f"❌ Toujours pas assez d'espace. Libérez manuellement ~{required_space_gb - free_gb:.1f} GB")
        print("   Conseil: cleanup_huggingface_cache(delete=True)")
        return False


# Raccourci pour import rapide
def quick_cleanup():
    """Nettoyage rapide - à exécuter en début de notebook."""
    return full_cleanup(
        base_dir=".",
        delete_hf_cache=False,
        delete_torch_cache=False,
        delete_old_embeddings=False,
        verbose=True
    )


if __name__ == "__main__":
    # Test standalone
    full_cleanup()
