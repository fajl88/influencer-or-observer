#!/usr/bin/env python
"""
Script de vérification de l'installation
Vérifie que toutes les dépendances sont installées et que la structure est correcte
"""

import sys
from pathlib import Path

def check_python_version():
    """Vérifie la version de Python"""
    print("🐍 Vérification de la version Python...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro} (Requis: 3.8+)")
        return False

def check_imports():
    """Vérifie que les packages principaux sont installés"""
    print("\n📦 Vérification des packages...")
    
    packages = [
        ('pandas', 'Manipulation de données'),
        ('numpy', 'Calculs numériques'),
        ('sklearn', 'Machine Learning'),
        ('nltk', 'NLP'),
        ('matplotlib', 'Visualisation'),
        ('seaborn', 'Visualisation'),
        ('transformers', 'Transformers (HuggingFace)'),
        ('torch', 'PyTorch'),
        ('lightgbm', 'LightGBM'),
    ]
    
    all_ok = True
    for package, description in packages:
        try:
            __import__(package)
            print(f"   ✓ {package:<15} - {description}")
        except ImportError:
            print(f"   ✗ {package:<15} - {description} (MANQUANT)")
            all_ok = False
    
    return all_ok

def check_nltk_data():
    """Vérifie que les données NLTK sont téléchargées"""
    print("\n📚 Vérification des données NLTK...")
    try:
        from nltk.corpus import stopwords
        french_stopwords = stopwords.words('french')
        print(f"   ✓ Stopwords français disponibles ({len(french_stopwords)} mots)")
        return True
    except Exception as e:
        print(f"   ✗ Stopwords français non disponibles")
        print(f"      Exécutez: python -c \"import nltk; nltk.download('stopwords')\"")
        return False

def check_directory_structure():
    """Vérifie la structure des dossiers"""
    print("\n📁 Vérification de la structure du projet...")
    
    required_dirs = [
        'data',
        'notebooks',
        'src',
        'models',
        'submissions',
        'reports',
        'reports/figures',
        'config'
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print(f"   ✓ {dir_path}/")
        else:
            print(f"   ✗ {dir_path}/ (MANQUANT)")
            all_ok = False
    
    return all_ok

def check_data_files():
    """Vérifie la présence des fichiers de données"""
    print("\n📊 Vérification des fichiers de données...")
    
    data_files = [
        'data/train.jsonl',
        'data/kaggle_test.jsonl'
    ]
    
    all_ok = True
    for file_path in data_files:
        path = Path(file_path)
        if path.exists() and path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"   ✓ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"   ✗ {file_path} (MANQUANT)")
            all_ok = False
    
    return all_ok

def check_config_files():
    """Vérifie la présence des fichiers de configuration"""
    print("\n⚙️  Vérification des fichiers de configuration...")
    
    config_files = [
        'config/config.json',
        'requirements.txt',
        'README.md'
    ]
    
    all_ok = True
    for file_path in config_files:
        path = Path(file_path)
        if path.exists() and path.is_file():
            print(f"   ✓ {file_path}")
        else:
            print(f"   ✗ {file_path} (MANQUANT)")
            all_ok = False
    
    return all_ok

def check_src_modules():
    """Vérifie que les modules src sont importables"""
    print("\n🔧 Vérification des modules src/...")
    
    modules = [
        'src.data_loader',
        'src.preprocessing',
        'src.feature_engineering',
        'src.models',
        'src.evaluation',
        'src.utils'
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            module_name = module.split('.')[-1]
            print(f"   ✓ {module_name}.py")
        except ImportError as e:
            module_name = module.split('.')[-1]
            print(f"   ✗ {module_name}.py (ERREUR: {e})")
            all_ok = False
    
    return all_ok

def main():
    """Fonction principale"""
    print("="*60)
    print("VÉRIFICATION DE L'INSTALLATION".center(60))
    print("Projet: Influencer or Observer".center(60))
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Packages Python", check_imports),
        ("Données NLTK", check_nltk_data),
        ("Structure Dossiers", check_directory_structure),
        ("Fichiers de Données", check_data_files),
        ("Fichiers de Config", check_config_files),
        ("Modules Source", check_src_modules),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n⚠️  Erreur lors de la vérification de '{name}': {e}")
            results[name] = False
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ".center(60))
    print("="*60)
    
    all_passed = all(results.values())
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✓ OK" if result else "✗ ERREUR"
        print(f"{name:<25} : {status}")
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 Toutes les vérifications sont passées!")
        print("✨ Vous êtes prêt à commencer le projet!")
        print("\n📖 Prochaines étapes:")
        print("   1. Consultez README.md ou QUICKSTART.md")
        print("   2. Lancez Jupyter: jupyter notebook")
        print("   3. Ouvrez notebooks/01_EDA.ipynb ou notebooks/02_baseline.ipynb")
    else:
        print(f"\n⚠️  {total - passed}/{total} vérifications ont échoué")
        print("📝 Corrigez les erreurs ci-dessus avant de continuer")
        print("\n💡 Conseils:")
        print("   - Installez les packages manquants: pip install -r requirements.txt")
        print("   - Téléchargez les stopwords: python -c \"import nltk; nltk.download('stopwords')\"")
        print("   - Vérifiez que vous êtes dans le bon répertoire")
    
    print("\n" + "="*60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
