#!/usr/bin/env python3
"""tools/lost_lines.py BASE OURS MERGED [N] - every non-trivial line OURS added since BASE (git diff BASE..OURS) must still be in
MERGED's copy of that file.  Prints the missing ones per file.  (tools/sync-upstream.sh runs this after every merge.)"""
import subprocess, sys, collections
base, ours, merged = sys.argv[1:4]
diff = subprocess.run(['git', 'diff', '--no-color', '-U0', '--no-renames', f'{base}..{ours}'], capture_output=True, text=True, errors='replace').stdout
added = collections.defaultdict(list); f = None
for line in diff.splitlines():
    if line.startswith('+++ '):
        f = line[6:] if line.startswith('+++ b/') else None
    elif line.startswith('+') and not line.startswith('+++') and f:
        t = line[1:].strip()
        if len(t) >= 12 and not t.startswith(('//', '#', '/*', '*', '"""')) and t not in ('{', '}', 'else:', 'return', 'pass', 'break', 'continue'):
            added[f].append(t)
missing_total = 0; report = []
for f, lines in sorted(added.items()):
    r = subprocess.run(['git', 'show', f'{merged}:{f}'], capture_output=True, text=True, errors='replace')
    if r.returncode != 0:
        report.append((f, len(lines), ['<FILE MISSING IN MERGE>'])); missing_total += len(lines); continue
    have = set(l.strip() for l in r.stdout.splitlines())
    miss = [l for l in lines if l not in have]
    if miss: report.append((f, len(lines), miss)); missing_total += len(miss)
total = sum(len(v) for v in added.values())
print(f'{total} added lines checked in {len(added)} files; {missing_total} missing in {len(report)} files')
for f, n, miss in sorted(report, key=lambda r: -len(r[2])):
    print(f'== {f}: {len(miss)}/{n} missing')
    for m in miss[:int(sys.argv[4]) if len(sys.argv) > 4 else 6]: print('   ', m[:160])
