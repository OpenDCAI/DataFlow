# hooks/post_gen_project.py
# Documentation see: https://cookiecutter.readthedocs.io/en/stable/advanced/hooks.html

import os
import shutil

def remove(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)

# include examples
include_examples = "{{ cookiecutter.include_examples }}"

if include_examples != "yes":
    print("Removing example operators, pipelines, and tests...")

    paths_to_remove = [
        "operators/example_operator.py",
        "pipelines/example_pipeline.py",
        "tests/test_example.py",
    ]

    for p in paths_to_remove:
        if os.path.exists(p):
            remove(p)


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
