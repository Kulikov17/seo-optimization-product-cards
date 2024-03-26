import torch
import torch.optim as optim
import pandas as pd
from tqdm.auto import tqdm
from src.ml.transforms import get_train_transform, get_valid_transform
from src.ml.dataset import WildberriesDataset
from src.ml.loss import LabelSmoothingCrossEntropy
from src.ml.models import build_model


def train(model, loader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for _, data in tqdm(enumerate(loader), total=len(loader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation.
        loss.backward()
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(loader.dataset))
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(loader.dataset))
    return epoch_loss, epoch_acc


def save_model(dir_path, model, optimizer):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f"{dir_path}/model.pth")


def training(dir_path,
             epochs=3,
             lr=1e-3,
             batch_size=8,
             image_size=224,
             num_workers=2):

    work_path = f'{dir_path}/wildberries'

    train_ann = pd.read_csv(f'{work_path}/train_ann.csv')
    val_ann = pd.read_csv(f'{work_path}/val_ann.csv')

    train_set = WildberriesDataset(work_path,
                                   train_ann,
                                   transform=get_train_transform(image_size))

    val_set = WildberriesDataset(work_path,
                                 val_ann,
                                 transform=get_valid_transform(image_size))

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=True)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build the model
    model = build_model(num_classes=2).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Loss function
    criterion = LabelSmoothingCrossEntropy()

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='max',
                                                     patience=3,
                                                     threshold=0.9,
                                                     min_lr=1e-6,
                                                     verbose=True)

    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_loss, train_acc = train(model,
                                      train_loader,
                                      optimizer,
                                      criterion,
                                      device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train loss: {train_loss:.3f}, train acc: {train_acc:.3f}")
        print(f"Val loss: {val_loss:.3f}, val acc: {val_acc:.3f}")
        print('-'*50)

        if scheduler is not None:
            scheduler.step(val_acc)

    save_model(dir_path, model, optimizer)

    print('TRAINING COMPLETE')
