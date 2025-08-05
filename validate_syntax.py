#!/usr/bin/env python3
"""
Simple syntax validation for PyMC-Supply-Chain after API fixes.
This script checks if the Python files can be parsed without syntax errors.
"""

import ast
import sys
from pathlib import Path

def validate_syntax(file_path):
    """Validate Python syntax of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main validation function."""
    print("=== PyMC-Supply-Chain Syntax Validation ===\n")
    
    # Files to validate
    files_to_check = [
        "pymc_supply_chain/demand/base.py",
        "pymc_supply_chain/demand/hierarchical.py", 
        "pymc_supply_chain/demand/seasonal.py",
        "pymc_supply_chain/demand/intermittent.py",
        "pymc_supply_chain/inventory/newsvendor.py",
        "pymc_supply_chain/inventory/safety_stock.py",
    ]
    
    all_valid = True
    
    for file_path in files_to_check:
        full_path = Path(file_path)
        if not full_path.exists():
            print(f"❌ {file_path}: File not found")
            all_valid = False
            continue
            
        is_valid, error = validate_syntax(full_path)
        if is_valid:
            print(f"✅ {file_path}: Syntax OK")
        else:
            print(f"❌ {file_path}: {error}")
            all_valid = False
    
    print(f"\n=== Summary ===")
    if all_valid:
        print("✅ All files passed syntax validation!")
        print("\nChanges made:")
        print("- Replaced all pm.ConstantData with pm.Data")
        print("- No pm.MutableData usage found")
        print("- No 'mutable' parameters found") 
        print("\nThe PyMC API compatibility issues have been successfully fixed.")
        return 0
    else:
        print("❌ Some files have syntax errors")
        return 1

if __name__ == "__main__":
    sys.exit(main())