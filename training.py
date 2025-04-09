import math
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import f1_score, accuracy_score, recall_score


def progress(loss, epoch, batch, batch_size, dataset_size):
    """
    Print the progress of the training for each epoch
    """
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()


def train_dataset(_epoch, dataloader, model, loss_function, optimizer):
    # IMPORTANT: switch to train mode
    # enable regularization layers, such as Dropout
    model.train()
    running_loss = 0.0

    # obtain the model's device ID
    device = next(model.parameters()).device

    for index, batch in enumerate(dataloader, 1):
        # get the inputs (batch)
        inputs, labels, lengths = batch

        # move the batch tensors to the right device
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device) # EX9

        # Step 1 - zero the gradients
        # Remember that PyTorch accumulates gradients.
        # We need to clear them out before each batch!
        optimizer.zero_grad() # EX9

        # Step 2 - forward pass: y' = model(x)
        if model.__class__.__name__ in ['BaselineDNN', 'LSTM']:
            outputs = model(inputs, lengths)
        else:
            outputs = model(inputs)

        # fix label type for different loss func
        # print(loss_function.__dict__)
        # if loss_function.__func__.__name__ == 'BCEWithLogitsLoss':
        #     labels = torch.nn.functional.one_hot(labels, num_classes=2)

        # Step 3 - compute loss: L = loss_function(y, y')
        try:
            loss = loss_function(outputs, labels)
        except ValueError:
            # fix labels for 'BCEWithLogitsLoss' loss function
            bin_labels = torch.nn.functional.one_hot(labels.long(), num_classes=2)
            loss = loss_function(outputs, bin_labels.float())

        # Step 4 - backward pass: compute gradient wrt model parameters
        loss.backward()  # EX9

        # Step 5 - update weights
        optimizer.step()  # EX9

        running_loss += loss.data.item()

        # print statistics
        progress(loss=loss.data.item(),
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))

    return running_loss / index


def eval_dataset(dataloader, model, loss_function):
    # IMPORTANT: switch to eval mode
    # disable regularization layers, such as Dropout
    model.eval()
    running_loss = 0.0

    y_pred = []  # the predicted labels
    y = []  # the gold labels

    # obtain the model's device ID
    device = next(model.parameters()).device

    # IMPORTANT: in evaluation mode, we don't want to keep the gradients
    # so we do everything under torch.no_grad()
    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
            # get the inputs (batch)
            inputs, labels, lengths = batch

            # Step 1 - move the batch tensors to the right device
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device) # EX9

            # Step 2 - forward pass: y' = model(x)
            if model.__class__.__name__ in ['BaselineDNN', 'LSTM']:
                outputs = model(inputs, lengths)
            else:
                outputs = model(inputs)

            # fix label type for different loss func
            # print(loss_function.__dict__)
            # if loss_function.__func__.__name__ == 'BCEWithLogitsLoss':
            #     labels = torch.nn.functional.one_hot(labels, num_classes=2)

            # Step 3 - compute loss: L = loss_function(y, y')
            try:
                loss = loss_function(outputs, labels)
            except ValueError:
                # fix labels for 'BCEWithLogitsLoss' loss function
                bin_labels = torch.nn.functional.one_hot(labels.long(), num_classes=2)
                loss = loss_function(outputs, bin_labels.float())

            # Step 4 - make predictions (class = argmax of posteriors)
            if outputs.shape[1] == 1:  # Binary classification (equivalent to 0.5 threshold)
                predictions = (outputs > 0.5).long().squeeze()
            else:  # Multi-class classification
                _, predictions = torch.max(outputs, 1) # EX9

            # Step 5 - collect the predictions, gold labels and batch loss
            y_pred.extend(predictions.cpu().numpy())
            y.extend(labels.cpu().numpy())  # EX9

            running_loss += loss.data.item()

    return running_loss / index, (y_pred, y)


def torch_train_val_split(
    dataset, batch_train, batch_eval, val_size=0.2, shuffle=True, seed=420
):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_size * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset, batch_size=batch_train, sampler=train_sampler, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(
        dataset, batch_size=batch_eval, sampler=val_sampler, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader


def get_metrics_report(y, y_hat):
    # Convert values to lists
    #y = np.concatenate(y, axis=0)
    #y_hat = np.concatenate(y_hat, axis=0)
    # report metrics
    report = f'  accuracy: {accuracy_score(y, y_hat)}\n  recall: ' + \
        f'{recall_score(y, y_hat, average="macro")}\n  f1-score: {f1_score(y, y_hat,average="macro")}'
    return report
