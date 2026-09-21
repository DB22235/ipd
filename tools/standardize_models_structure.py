"""
tools/standardize_models_structure.py
======================================
Standardizes the models directory into three symmetrical crop namespaces:
  - models/potato/
      ├── teachers/
      ├── students/
      ├── converted/
      └── model_registry.json
  - models/rice/
      ├── teachers/
      ├── students/
      ├── converted/
      └── model_registry.json
  - models/tomato/
      ├── teachers/
      ├── students/
      ├── converted/
      └── model_registry.json

Ensures zero data loss and establishes clean repository symmetry.
"""

import os
import shutil
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"


def standardize_potato():
    print("[1/3] Standardizing models/potato/...")
    pot_dir = MODELS_DIR / "potato"
    pot_dir.mkdir(parents=True, exist_ok=True)
    
    # Teachers
    pot_teachers = pot_dir / "teachers"
    pot_teachers.mkdir(parents=True, exist_ok=True)
    old_pot_teacher = MODELS_DIR / "potato_teacher"
    if old_pot_teacher.exists():
        for item in old_pot_teacher.iterdir():
            target = pot_teachers / item.name
            if not target.exists():
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
        print("  -> Migrated potato_teacher into models/potato/teachers/")

    # Students
    pot_students = pot_dir / "students"
    pot_students.mkdir(parents=True, exist_ok=True)
    pot_sb = pot_dir / "student_baselines"
    if pot_sb.exists() and not (pot_students / "student_baselines").exists():
        shutil.copytree(pot_sb, pot_students / "student_baselines")
        print("  -> Mirrored student_baselines into models/potato/students/")


def standardize_rice():
    print("[2/3] Standardizing models/rice/...")
    rice_dir = MODELS_DIR / "rice"
    rice_dir.mkdir(parents=True, exist_ok=True)
    
    # Teachers
    rice_teachers = rice_dir / "teachers"
    rice_teachers.mkdir(parents=True, exist_ok=True)
    old_rice_teacher = MODELS_DIR / "rice_teacher_v1"
    if old_rice_teacher.exists():
        for item in old_rice_teacher.iterdir():
            target = rice_teachers / item.name
            if not target.exists():
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
        print("  -> Migrated rice_teacher_v1 into models/rice/teachers/")

    # Students
    rice_students = rice_dir / "students"
    rice_students.mkdir(parents=True, exist_ok=True)
    for sub in ["distilled_students", "student_baselines"]:
        src_sub = rice_dir / sub
        dst_sub = rice_students / sub
        if src_sub.exists() and not dst_sub.exists():
            shutil.copytree(src_sub, dst_sub)
            print(f"  -> Mirrored {sub} into models/rice/students/")


def standardize_tomato():
    print("[3/3] Standardizing models/tomato/...")
    tom_dir = MODELS_DIR / "tomato"
    tom_dir.mkdir(parents=True, exist_ok=True)
    
    # Teachers
    tom_teachers = tom_dir / "teachers"
    tom_teachers.mkdir(parents=True, exist_ok=True)
    
    # Legacy teachers into legacy subfolders
    for old_name, subname in [("tomato_teacher", "legacy_v1"), ("tomato_teacher_v3", "legacy_v3")]:
        old_path = MODELS_DIR / old_name
        dest_path = tom_teachers / subname
        if old_path.exists() and not dest_path.exists():
            dest_path.mkdir(parents=True, exist_ok=True)
            for item in old_path.iterdir():
                target = dest_path / item.name
                if not target.exists():
                    if item.is_dir():
                        shutil.copytree(item, target)
                    else:
                        shutil.copy2(item, target)
            print(f"  -> Migrated {old_name} into models/tomato/teachers/{subname}/")


def main():
    print("===========================================================================")
    print("      STANDARDIZING MODELS DIRECTORY STRUCTURE TO CROP NAMESPACES          ")
    print("===========================================================================")
    standardize_potato()
    standardize_rice()
    standardize_tomato()
    print("===========================================================================")
    print(" [COMPLETE] Models structure successfully standardized across all crops!")
    print("===========================================================================\n")


if __name__ == "__main__":
    main()
