import json
import sys

def convert_py_to_ipynb(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Разделитель для ячеек
    blocks = content.split('#%%')
    
    cells = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        if block.startswith('[markdown]'):
            cell_type = 'markdown'
            # Убираем метку и начальные '# '
            lines = block.split('\n')[1:]
            source = []
            for line in lines:
                if line.startswith('# '):
                    source.append(line[2:] + '\n')
                elif line.startswith('#'):
                    source.append(line[1:] + '\n')
                else:
                    source.append(line + '\n')
                    
        elif block.startswith('[code]'):
            cell_type = 'code'
            lines = block.split('\n')[1:]
            source = [line + '\n' for line in lines]
            
        else:
            # по умолчанию
            cell_type = 'code'
            source = [line + '\n' for line in block.split('\n')]
            
        # Удаляем последний перенос у последней строки ячейки для аккуратности
        if source and source[-1].endswith('\n'):
            source[-1] = source[-1][:-1]
            
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source
        }
        if cell_type == 'code':
            cell["outputs"] = []
            cell["execution_count"] = None
            
        cells.append(cell)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Успешно конвертировано: {input_file} -> {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_ipynb.py <input.py> <output.ipynb>")
        sys.argv = ['convert_to_ipynb.py', 'kaggle_my_train_odin.py', 'kaggle_my_train_odin.ipynb']
    
    convert_py_to_ipynb(sys.argv[1], sys.argv[2])
