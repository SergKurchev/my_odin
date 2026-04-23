import json
import os

def convert_py_to_ipynb(py_path, ipynb_path):
    with open(py_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cells = []
    current_cell_type = None
    current_cell_content = []

    for line in lines:
        if line.startswith('#%% [markdown]'):
            if current_cell_type:
                cells.append({
                    "cell_type": current_cell_type,
                    "metadata": {},
                    "source": current_cell_content
                })
            current_cell_type = "markdown"
            current_cell_content = []
        elif line.startswith('#%% [code]'):
            if current_cell_type:
                cells.append({
                    "cell_type": current_cell_type,
                    "metadata": {},
                    "source": current_cell_content
                })
            current_cell_type = "code"
            current_cell_content = []
        elif line.startswith('#%%'):
             # Generic cell marker
            if current_cell_type:
                cells.append({
                    "cell_type": current_cell_type,
                    "metadata": {},
                    "source": current_cell_content
                })
            current_cell_type = "code"
            current_cell_content = []
        else:
            current_cell_content.append(line)

    if current_cell_type:
        cells.append({
            "cell_type": current_cell_type,
            "metadata": {},
            "source": current_cell_content
        })

    # Fix code cells: remove the comment markers from the beginning if they were caught in the content
    # (though my logic above avoids that)
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

if __name__ == "__main__":
    py_file = r"c:\Users\NeverGonnaGiveYouUp\OneDrive\Рабочий стол\study_materials\Skoltech\projects\StrawPick\NBV_article\my_ODIN_for_strawberry\odin\kaggle_my_train_odin.py"
    ipynb_file = r"c:\Users\NeverGonnaGiveYouUp\OneDrive\Рабочий стол\study_materials\Skoltech\projects\StrawPick\NBV_article\my_ODIN_for_strawberry\odin\kaggle_my_train_odin.ipynb"
    convert_py_to_ipynb(py_file, ipynb_file)
    print(f"Converted {py_file} to {ipynb_file}")
