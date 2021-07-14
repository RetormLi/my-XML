from collections import defaultdict
import json

import click


@click.command()
@click.argument('label_dict', type=click.File('w'))
@click.argument('data_files', type=click.File('r'), nargs=-1)
def build_label_dict_from_data_file_order_by_frequency(
        label_dict, data_files
):
    labels = defaultdict(int)
    for data_file in data_files:
        for _ in data_file:
            label_line = data_file.readline()
            for label in label_line.strip().split():
                labels[label] += 1
    # 排序
    label_indices = sorted(labels, key=lambda x: labels[x])
    results = {
        label: label_index
        for label_index, label in enumerate(label_indices)
    }
    # 输出
    print(len(results))
    json.dump(results, label_dict, indent=2)


if __name__ == '__main__':
    build_label_dict_from_data_file_order_by_frequency()