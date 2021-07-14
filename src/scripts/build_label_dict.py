import json

import click


@click.command('Build Label Dict')
@click.argument('label_dict', type=click.File('w'))
@click.argument('label_files', type=click.File('r'), nargs=-1)
def build_label_dict(label_dict, label_files):
    labels = {}
    for label_file in label_files:
        for line in label_file.readlines():
            for label in line.strip().split():
                if label not in labels:
                    labels[label] = len(labels)
    print(len(labels))
    json.dump(labels, label_dict, indent=2)


if __name__ == '__main__':
    build_label_dict()
