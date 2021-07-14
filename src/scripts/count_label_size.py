import click


@click.command('Count Label Classes')
@click.argument('label_files', type=click.File('r'), nargs=-1)
def count_label_size(label_files):
    labels = set()
    for label_file in label_files:
        for line in label_file.readlines():
            labels = labels.union(line.strip().split())
    print(len(labels))


if __name__ == '__main__':
    count_label_size()
