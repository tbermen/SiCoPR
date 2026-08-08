import json, os, sys

with open('com_functions/registry.json', encoding='utf-8-sig') as f:
    reg = json.load(f)

fns = reg['functions']
not_verified = [fn['name'] for fn in fns if fn.get('status') != 'verified']
missing_impl = []
for fn in fns:
    path = os.path.join('com_functions', 'fn', fn['name'], 'py_impl.py')
    if not os.path.exists(path):
        missing_impl.append(fn['name'])

print(f'Total functions in registry: {len(fns)}')
print(f'Not verified: {len(not_verified)}')
if not_verified:
    for n in not_verified[:30]:
        print(f'  - {n}')
    if len(not_verified) > 30:
        print(f'  ... and {len(not_verified)-30} more')
print(f'Missing py_impl.py: {len(missing_impl)}')
if missing_impl:
    for n in missing_impl[:30]:
        print(f'  - {n}')

if not_verified or missing_impl:
    sys.exit(1)
print('OK: All functions verified and py_impl.py present.')
