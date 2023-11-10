import torch
import torchvision
import numpy
import copy
import tqdm

from torch.utils.data import Dataset, DataLoader

class LeNet5(torch.nn.Module):
    def __init__(self, input_channels):
        super(LeNet5, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=input_channels, out_channels=6, kernel_size=5, stride=1)
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=84)
        self.fc3 = torch.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return torch.nn.functional.cross_entropy(input, target)


class Server:
    def __init__(self, model, client_params):
        self.model = copy.deepcopy(model)
        self.client_params = client_params
        self.n_client = len(self.client_params)
        self.parameters = self.client_params[0]

        self.fed_avg()
        self.model.load_state_dict(self.parameters)

    def fed_avg(self):
        for client in range(1, self.n_client):
            for key in self.parameters:
                new_params = self.client_params[client][key]
                # print(new_params.equal(self.server_params[key]), end=' | ')
                self.parameters[key] = self.parameters[key].add(
                    new_params)
                # tmp_1 = copy.deepcopy(new_params)
                # tmp_2 = copy.deepcopy(self.parameters[key])
                # print(new_params.equal(tmp_2.div(2)))
        for key in self.parameters:
            self.parameters[key] = self.parameters[key].div(2)


class DealDataset(Dataset):
    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = idx
        self.len = len(idx)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img, target = self.dataset[self.idx[index]]
        return img, target

def idx_split(dataset, mode='iid', n_dataset=1, n_data_each_set=1):
    labels_list = dataset.targets.tolist()
    all_labels = set(labels_list)
    idx_label = dict()
    for label in all_labels:
        idx_label[label] = list()
        for idx, label_in_list in enumerate(labels_list):
            if label_in_list == label:
                idx_label[label] += [idx]

    if mode == 'iid':
        if n_dataset * n_data_each_set > len(dataset):
            raise ValueError(
                f'number of client ({n_dataset}) times number of data of each client ({n_data_each_set}) no more than number of total data ({len(dataset)})')
        n_each_set = dict()
        for label in all_labels:
            n_each_set[label] = int(
                len(idx_label[label]) / len(labels_list) * n_data_each_set)
        dataset_splited = dict()
        left_idx_label = idx_label
        for i in range(n_dataset):
            dataset_splited[i] = list()
            for label in all_labels:
                # print(len(left_idx_label[label]), n_each_set[label])
                choiced_idx = numpy.random.choice(
                    left_idx_label[label],
                    n_each_set[label],
                    replace=False)
                dataset_splited[i] += list(choiced_idx)
                left_idx_label[label] = list(
                    set(left_idx_label[label]) - set(dataset_splited[i]))
                # print(f'client={i}, label={label}, dataset_splited[i]={len(dataset_splited[i])}')
        return dataset_splited
    elif mode == 'non-iid':
        print('TO DO.')


def train_model(model, dataset, device='cpu', epochs=1, tqdm_position=0):
    trained_model = copy.deepcopy(model).to(device)
    trained_model.train()
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trained_model.parameters())
    for epoch in range(epochs):
        for i, (data, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = trained_model(data.to(device))
            loss = criterion(output, label.to(device))
            loss.backward()
            optimizer.step()

        #     if (i+1) % 100 == 0:
        #         print('\r', end='')
        #         print(
        #             f'step [{i+1}/{len(train_dataloader)}], loss: {loss.item():.4f}', end='')
        # print(f'\nepoch {epoch+1}/{epochs} down.')
    return trained_model


def eval_model(model, dataset, device):
    server_model = copy.deepcopy(model)
    server_model.eval()
    server_model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for images, labels in data_loader:
            outputs = server_model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
        print('Test Accuracy: {:.2f}%'.format(100 * correct / total))

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True)
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=torchvision.transforms.ToTensor())

n_total_client = 5
n_data = 12000
communication_round = 5
epochs = 1
n_client = 2

model = LeNet5(input_channels=1)
idx_splited = idx_split(dataset=train_dataset,
                        n_dataset=n_total_client,
                        n_data_each_set=n_data)
# print(len(idx_splited), len(idx_splited[0]))
dataset_client = dict()
for i in range(n_total_client):
    dataset_client[i] = DealDataset(train_dataset, idx_splited[i])

server_model = copy.deepcopy(model)
tqdm_position = 0
for i in tqdm.tqdm(range(communication_round),
                   desc='communication_round',
                   position=tqdm_position):
    client = dict()
    client_param = dict()
    choicen_client = numpy.random.choice(
        range(n_total_client), n_client, replace=False)
    for j, k in enumerate(choicen_client):
        client[j] = train_model(
            model=server_model,
            dataset=dataset_client[k],
            device=device,
            epochs=epochs,
            tqdm_position=tqdm_position+1)
        client_param[j] = client[j].state_dict()
    server_model = Server(model=model, client_params=client_param).model

eval_model(client[0], test_dataset, device)
eval_model(client[1], test_dataset, device)
eval_model(server_model, test_dataset, device)

# trained_model = copy.deepcopy(model).to(device)
# # trained_model = LeNet5(input_channels=1).to(device)
# trained_model.train()
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(trained_model.parameters())
# for epoch in range(1):
#     for i, (data, label) in enumerate(train_dataloader):
#         optimizer.zero_grad()
#         output = trained_model(data.to(device))
#         loss = criterion(output, label.to(device))
#         loss.backward()
#         optimizer.step()

#         if (i+1) % 100 == 0:
#             print('\r', end='')
#             print(
#                 f'step [{i+1}/{len(train_dataloader)}], loss: {loss.item():.4f}', end='')
#     print(f'\nepoch {epoch+1}/{1} down.')
# eval_model(trained_model, test_dataset)