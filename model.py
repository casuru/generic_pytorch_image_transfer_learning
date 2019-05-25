import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import models, transforms, datasets
import argparse


def get_model_criterion_and_optimizer(name, num_classes, learning_rate = 0.0001,
                                      device = "cuda" if torch.cuda.is_available() else "cpu"):


    if name == "resnet50":

        model = models.resnet50(pretrained = True)

    elif name == "vgg16":

        model = models.vgg16(pretrained = True)

    elif name == "vgg19":

        model = models.vgg19(pretrained = True)

    else:

        raise ValueError("Base model: {model} is not supported".format(model = name))
    
    
    for param in model.parameters():

        param.require_grad = False


    if name.startswith("resnet"):

        num_attrs = model.fc.in_features
        model.fc = nn.Linear(num_attrs, num_classes)


    if name.startswith("vgg"):

        num_attrs = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(num_attrs, num_classes)

    model.to(device)



    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    return model, criterion, optimizer


def get_dataloaders(training_folder, validation_folder = None, batch_size = 128):

    image_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224, )),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    training_set = datasets.ImageFolder(training_folder, transform = image_transform)
    training_loader = data.DataLoader(training_set, batch_size = batch_size, shuffle = True)


    if validation_folder is None:

        return {"train": training_loader}

    validation_set = datasets.ImageFolder(validation_folder, transform = image_transform)
    validation_loader = data.DataLoader(validation_set, batch_size = batch_size, shuffle = True)


    return {"train": training_loader, "val": validation_loader}

def train_and_save_weights(model, criterion, optimizer, dataloaders, 
                           num_epochs = 10, device = "cuda" if torch.cuda.is_available() else "cpu"):


    for epoch in range(num_epochs):

        num_batches = 0.0
        best_accuracy = 0.0
        training_loss = 0.0
        training_accuracy = 0.0
        validation_loss = 0.0
        validation_accuracy = 0.0

        model.train()
        for x_batch, y_batch in dataloaders['train']:

            num_batches += 1

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)

            optimizer.zero_grad()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            batch_accuracy = (torch.argmax(outputs, 1) == y_batch).sum().item() / len(y_batch)

            training_loss += loss.item()
            training_accuracy += batch_accuracy

        training_loss /= num_batches
        training_accuracy /= num_batches

        print("Train acc", training_accuracy)

        if "val" in dataloaders:

            model.eval()
            with torch.no_grad():

                for x_batch, y_batch in dataloaders['val']:

                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    outputs = model(x_batch)

                    optimizer.zero_grad()
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                    batch_accuracy = (torch.argmax(outputs, 1) == y_batch).sum().item() / len(y_batch)
                    validation_loss += loss.item()
                    validation_accuracy += batch_accuracy

                validation_loss /= num_batches
                validation_accuracy /= num_batches

                print("Val acc", validation_accuracy)

        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), str(datetime.datetime.now()) + ".pth")
        


            


            





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("num_classes", type=int)
    parser.add_argument("learning_rate", type=float)
    args = parser.parse_args()

    model, criterion, optimizer = get_model_criterion_and_optimizer(args.model_name, 
                                                                    args.num_classes, learning_rate = args.learning_rate)

    print(model)
