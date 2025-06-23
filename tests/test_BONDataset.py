from ACAgraphML.Dataset import ZINC_Dataset
from ACAgraphML.Datasets.BONDataset import BONDataset


def test_bon_dataset():
    dataset = ZINC_Dataset.SMALL_TEST.load()
    bon_dataset = BONDataset(dataset)

    # Assert the shape of BON features
    assert len(bon_dataset) == len(dataset)
    bon, steadyBon, boe, y = bon_dataset[0]
    assert bon.shape == (bon_dataset.nNodeFeats,)
    assert steadyBon.shape == (bon_dataset.nNodeFeats,)
    assert boe.shape == (bon_dataset.nEdgeFeats - 1,)
    assert y.shape == (1,)
