import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import train_dataset, val_dataset
from model import Image_Classification, weights_init
import os
from ray import tune


def train(config, weight_update, checkpoint_dir=None, epochs=400):
    model = Image_Classification(config["l1"], config["l2"])
    model.apply(weights_init)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    if weight_update=='generalized_delta':
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    if weight_update=='delta':
        optimizer = optim.SGD(model.parameters(), lr=config["lr"])

    if weight_update=='adam':
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])


    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True, num_workers=8)




    for epoch in range(epochs):
        count = 0
        avg_train_loss = 0

        model.train()
        for batch_idx, (image, label) in enumerate(train_loader):
            count += 1

            image = image.to(device=device)
            label = label.to(device=device)

            out = model(image)
            loss = criterion(out, label.long())

            avg_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        avg_train_loss = avg_train_loss/count
        print(f'Epochs:{epoch+1}, Average Train Loss:{avg_train_loss}')


        count = 0
        avg_val_loss = 0
        total = 0
        correct = 0
        model.eval()
        for batch_idx, (image, label) in enumerate(val_loader):
            with torch.no_grad():
                count += 1

                image = image.to(device=device)
                label = label.to(device=device)

                out = model(image)
                preds = torch.argmax(out)
                total += 1
                correct += (preds == label)

                loss = criterion(out, label.long())

                avg_val_loss += loss.item()

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(avg_val_loss / count), accuracy=correct / total)

    print('Finished Training')

        
