"""
FIX #1: Action Smoothing Train-Test Mismatch

This script applies the fix for Issue #1 by modifying standing_env.py
to NOT reset prev_action during episode reset.

BACKUP: Creates a backup of the original file before modification.
"""

import os
import sys
import shutil
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
TARGET_FILE = os.path.join(PROJECT_ROOT, 'src', 'environments', 'standing_env.py')


def apply_fix():
    """Apply the action smoothing fix"""
    
    print("="*60)
    print("FIX #1: Action Smoothing Train-Test Mismatch")
    print("="*60)
    print()
    
    if not os.path.exists(TARGET_FILE):
        print(f"✗ Target file not found: {TARGET_FILE}")
        return False
    
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = TARGET_FILE.replace('.py', f'_backup_{timestamp}.py')
    shutil.copy2(TARGET_FILE, backup_file)
    print(f"✓ Backup created: {backup_file}")
    
    # Read original file
    with open(TARGET_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find and modify the line
    modified = False
    new_lines = []
    
    for i, line in enumerate(lines):
        # Look for the problematic line in reset() method
        if 'self.prev_action[:] = 0.0' in line and not line.strip().startswith('#'):
            # Comment it out
            indent = len(line) - len(line.lstrip())
            new_line = ' ' * indent + '# FIXED: Do not reset prev_action (maintains train-test consistency)\n'
            new_line += ' ' * indent + '# self.prev_action[:] = 0.0  # ← OLD CODE (causes inference failure)\n'
            new_lines.append(new_line)
            modified = True
            print(f"✓ Line {i+1} modified:")
            print(f"  OLD: {line.rstrip()}")
            print(f"  NEW: # {line.strip()}  # ← Commented out")
        else:
            new_lines.append(line)
    
    if not modified:
        print("✗ Could not find 'self.prev_action[:] = 0.0' in reset() method")
        print("  File may have already been modified or structure changed")
        return False
    
    # Write modified file
    with open(TARGET_FILE, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"\n✓ Fix applied successfully to: {TARGET_FILE}")
    print()
    print("WHAT CHANGED:")
    print("  - prev_action is NO LONGER reset to zeros during episode reset")
    print("  - This matches VecEnv behavior during training (where prev_action persists)")
    print("  - Agent will now see consistent action smoothing during inference")
    print()
    print("EXPECTED IMPACT:")
    print("  - Inference survival should increase from ~140 steps to 500+ steps")
    print("  - Agent behavior will be more consistent with training")
    print()
    print("NEXT STEPS:")
    print("  1. Re-run inference: python scripts/record_video.py --task standing --model <path>")
    print("  2. Run diagnostic: python scripts/diagnostics/test_action_smoothing.py")
    print("  3. If still failing, investigate Issues #2 and #3")
    print()
    
    return True


def revert_fix(backup_file):
    """Revert the fix using a backup file"""
    
    if not os.path.exists(backup_file):
        print(f"✗ Backup file not found: {backup_file}")
        return False
    
    shutil.copy2(backup_file, TARGET_FILE)
    print(f"✓ Reverted to backup: {backup_file}")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fix action smoothing train-test mismatch")
    parser.add_argument('--revert', type=str, help='Revert using specified backup file')
    args = parser.parse_args()
    
    if args.revert:
        revert_fix(args.revert)
    else:
        apply_fix()


if __name__ == "__main__":
    main()

