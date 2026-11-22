from src.training.preprocessing import split_dataset

# split_dataset only shuffles (path, label) pairs, so the paths can be fake.
IMBALANCED = [(f"clean_{i}.jpg", 0) for i in range(100)] + [(f"defect_{i}.jpg", 1) for i in range(400)]


def class_share(subset):
    return sum(label for _, label in subset) / len(subset)


def test_split_sizes_follow_the_ratios():
    train, val, test = split_dataset(IMBALANCED)
    assert (len(train), len(val), len(test)) == (350, 75, 75)


def test_split_keeps_the_class_balance_in_every_set():
    overall = class_share(IMBALANCED)
    for subset in split_dataset(IMBALANCED):
        assert abs(class_share(subset) - overall) < 0.02


def test_split_uses_every_image_exactly_once():
    train, val, test = split_dataset(IMBALANCED)
    assert sorted(train + val + test) == sorted(IMBALANCED)


def test_split_is_reproducible_with_the_same_seed():
    assert split_dataset(IMBALANCED, random_seed=1) == split_dataset(IMBALANCED, random_seed=1)
    assert split_dataset(IMBALANCED, random_seed=1) != split_dataset(IMBALANCED, random_seed=2)
