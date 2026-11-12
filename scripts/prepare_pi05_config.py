#!/usr/bin/env python3
"""Create an untracked local config with paths for this clone; opens no devices."""
import argparse
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=root / 'pi05/configs/pi05_realman_presets.local.toml')
args = parser.parse_args()
if args.output.exists():
    parser.error(f'{args.output} already exists; edit it or choose a new --output')
text = (root / 'pi05/configs/pi05_realman_presets.toml').read_text()
text = text.replace('/home/robot2/deltawam/src', str(root / 'fastwam/src'))
text = text.replace('/home/robot2/jhk/phywam-mope-real/real-records', str(root / 'recordings'))
text = text.replace('mode = "execute"', 'mode = "no_motion"')
text = re.sub(r'^enabled = true$', 'enabled = false', text, flags=re.MULTILINE)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(text)
print(f'Created {args.output}; execution.mode=no_motion; recording disabled.')
