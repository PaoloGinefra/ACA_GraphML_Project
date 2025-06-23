from ACAgraphML.Dataset import ZINC_Dataset
from ACAgraphML.Datasets.BONDataset import BONDataset


def test_bon_dataset():
    dataset = ZINC_Dataset.SMALL_TEST.load()
    bon_dataset = BONDataset(dataset)
    print(bon_dataset[0])
    assert True


if __name__ == "__main__":
    test_bon_dataset()
    print("BONDataset test passed successfully.")
