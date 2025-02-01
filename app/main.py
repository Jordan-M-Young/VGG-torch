"""Main Training Loop."""

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

import app.data as ad
import app.training as tr
import app.utils as ut
from app.vgg import VGGA


def main():
    """Main Training Loop."""
    TEST_FRACTION = 0.2
    BATCH_SIZE = 32
    EPOCHS = 2
    data = ad.load_data()

    images = data["images"]
    labels = data["labels"]

    dataset = ad.ImageDataset(images, labels)
    size = len(dataset)
    train_size = int((1 - TEST_FRACTION) * size)
    test_size = size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = VGGA()

    optimizer = Adam(params=model.parameters(), lr=0.0001)
    loss_fn = CrossEntropyLoss()

    for epoch in range(EPOCHS):
        # train
        tr_loss = tr.train(train_dataloader, model, optimizer, loss_fn)
        # validate
        ev_loss = tr.evaluate(test_dataloader, model, loss_fn)

        ut.log_epoch(epoch, tr_loss, ev_loss)


if __name__ == "__main__":
    main()
