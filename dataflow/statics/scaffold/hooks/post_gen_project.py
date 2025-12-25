# --- include_examples: simple & explicit ---
import os
import shutil

PROJECT_DIR = os.getcwd()
PACKAGE_NAME = "{{ cookiecutter.package_name }}"
INCLUDE_EXAMPLES = "{{ cookiecutter.include_examples }}"

def clear_dir(path):
    """Remove all contents under a directory, but keep the directory itself."""
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)

if INCLUDE_EXAMPLES != "yes":
    print("Clearing operators / pipelines / prompts directories...")

    base_dirs = [
        os.path.join(PROJECT_DIR, PACKAGE_NAME),
        os.path.join(PROJECT_DIR, "src", PACKAGE_NAME),
    ]

    for base in base_dirs:
        if not os.path.isdir(base):
            continue

        for sub in ("operators", "pipelines", "prompts"):
            target = os.path.join(base, sub)
            if os.path.isdir(target):
                clear_dir(target)
                print("  cleared:", os.path.relpath(target, PROJECT_DIR))


# License
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

license_choice = "{{ cookiecutter.license }}"
author = "{{ cookiecutter.author }}"

# --- handle LICENSE file ---
licenses_dir = "licenses"
target_license_file = "LICENSE"

if license_choice == "Proprietary":
    write_file(
        target_license_file,
        f"Copyright (c) {author}\n\nAll rights reserved.\n"
    )
else:
    license_template = os.path.join(licenses_dir, f"{license_choice}.txt")
    if not os.path.exists(license_template):
        raise RuntimeError(f"License template not found: {license_template}")

    content = read_file(license_template)
    write_file(target_license_file, content)

# remove licenses directory from generated project
if os.path.isdir(licenses_dir):
    shutil.rmtree(licenses_dir)

print(f"Applied license: {license_choice}")
