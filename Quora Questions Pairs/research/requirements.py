# requirements.py
import json
import ast
import sys
import re
from pathlib import Path
from importlib.metadata import distributions, packages_distributions

from IPython.core.magic import Magics, magics_class, line_magic

@magics_class
class MyMagics(Magics):

    @line_magic
    def updatereqs(self, line):
        """
        Usage: %updatereqs path/to/notebook.ipynb
        Scans the given notebook and merges its imports into requirements.txt.
        """
        line = line.strip()
        if not line:
            print(">>> Please provide the notebook path: %updatereqs notebook.ipynb")
            return

        nb_file = Path(line)
        if not nb_file.exists():
            print(f">>> File not found: {nb_file}")
            return

        print(f">>> Scanning: {nb_file.name}")

        # ---- Extract imports ----
        collected_modules = set()
        with open(nb_file, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        for cell in nb.get('cells', []):
            if cell.get('cell_type') != 'code':
                continue

            code = ''.join(cell.get('source', []))
            # Remove IPython magics and shell commands that break ast.parse
            clean_code = re.sub(r'^\s*[%!].*$', '', code, flags=re.MULTILINE)
            if not clean_code.strip():
                continue

            try:
                tree = ast.parse(clean_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            collected_modules.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            collected_modules.add(node.module.split('.')[0])
            except SyntaxError as e:
                # Optionally log which cell failed
                print(f">>>  Skipping cell with invalid Python syntax: {e}")
                continue

        # Filter stdlib
        try:
            stdlib = sys.stdlib_module_names
        except AttributeError:
            stdlib = set(sys.builtin_module_names)
            stdlib.update(['os', 'sys', 'json', 're', 'math', 'time', 'datetime',
                          'collections', 'itertools', 'functools', 'random', 'pathlib',
                          'logging', 'warnings', 'inspect', 'shutil', 'zipfile'])

        third_party = {p for p in collected_modules if p not in stdlib}
        print(f">>> Detected third-party imports: {sorted(third_party)}")

        # Map to distribution names
        import_to_dist = packages_distributions()
        dist_versions = {dist.metadata['Name']: dist.version for dist in distributions()}
        dist_names_lower = {name.lower(): name for name in dist_versions}

        new_packages = {}
        for imp in sorted(third_party):
            dist_names = import_to_dist.get(imp)
            if not dist_names:
                if imp.lower() in dist_names_lower:
                    dist_names = [dist_names_lower[imp.lower()]]
                elif imp in dist_versions:
                    dist_names = [imp]
                else:
                    print(f">>> Skipping '{imp}' (no matching package found)")
                    continue

            for dname in dist_names:
                if dname in dist_versions:
                    new_packages[dname] = dist_versions[dname]
                    break

        # Read existing requirements.txt
        existing = {}
        req_path = Path('requirements.txt')
        if req_path.exists():
            with open(req_path, 'r') as f:
                for line in f:
                    if '==' in line:
                        pkg, ver = line.strip().split('==', 1)
                        existing[pkg] = ver

        # Merge and write
        final = {**existing, **new_packages}
        with open(req_path, 'w') as f:
            for pkg, ver in sorted(final.items()):
                f.write(f"{pkg}=={ver}\n")

        added = set(new_packages) - set(existing)
        if added:
            print(f">>> Added: {', '.join(added)}")
        print(f">>> Total packages in requirements.txt: {len(final)}")


def load_ipython_extension(ipython):
    ipython.register_magics(MyMagics)