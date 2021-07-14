import click
import random


@click.command()
@click.argument('text', type=click.File('r'))
@click.argument('label', type=click.File('r'))
@click.argument('dev_ratio', type=click.FloatRange(0.0, 1.0))
@click.argument('train', type=click.File('w'))
@click.argument('train_label', type=click.File('w'))
@click.argument('dev', type=click.File('w'))
@click.argument('dev_label', type=click.File('w'))
def split_train_and_dev(text, label, dev_ratio: float,
                        train, train_label,
                        dev, dev_label):
    full = text.readlines()
    full_labels = label.readlines()
    assert len(full) == len(full_labels)
    length = len(full)

    dev_length = int(length * dev_ratio)
    dev_indices = random.sample(
        list(range(0, length)), dev_length
    )

    for index in range(0, length):
        if index in dev_indices:
            dev.write(full[index])
            dev_label.write(full_labels[index])
        else:
            train.write(full[index])
            train_label.write(full_labels[index])


if __name__ == '__main__':
    split_train_and_dev()
