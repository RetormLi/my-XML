from collections import defaultdict
import json
import click


@click.command()
@click.argument('label_dict', type=click.File('w'))
@click.argument('label_files', type=click.File('r'), nargs=-1)
def build_label_dict_order_by_frequency(
        label_dict, label_files
):
    labels = defaultdict(int)
    for label_file in label_files:
        for line in label_file.readlines():
            for label in line.strip().split():
                labels[label] += 1
    # 排序
    label_indices = sorted(labels, key=lambda x: labels[x])
    results = {
        label: label_index
        for label_index, label in enumerate(label_indices)
    }
    # 输出
    print(len(labels))
    json.dump(results, label_dict, indent=2)


if __name__ == '__main__':
    build_label_dict_order_by_frequency()