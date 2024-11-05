import torch, torchvision, datetime, os
import kernels
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

################################################# FUNCTIONS #################################################
def train_val_split(data):
    n_samples = len(datafolder)
    n_val = int(0.3 * n_samples)

    shuffled_indices = torch.randperm(n_samples)

    train_indices = shuffled_indices[:-n_val]
    val_indices = shuffled_indices[-n_val:]
    print(train_indices.shape, val_indices.shape, sep="\n")

    train_data = [datafolder[x] for x in train_indices]
    assert len(train_data) == train_indices.shape[0], "[*] ERROR: Something went wrong in splitting data (train data)"
    val_data = [datafolder[x] for x in val_indices]
    assert len(val_data) == val_indices.shape[0], "[*] ERROR: Something went wrong in splitting data (val data)"

    return train_data, val_data

def validate(model, val_loader) -> None:
    model.eval()
    for name, loader in [("knife", val_loader), ("scissor", val_loader), ("camera", val_loader), ("cellphone", val_loader), ("electronic", val_loader), ("laptop", val_loader), ("lighter", val_loader), ("powerbank", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():  # <1>
            for imgs, labels in loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f}".format(name , correct / total))
    return

# training loop with l2 regularization
def training_loop(n_epochs, optimizer, model, loss_fn,
                        train_loader) -> None:
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
        if epoch == 1 or epoch % 1 == 0:
            print(f"{datetime.datetime.now()} Epoch {epoch}, Training loss {loss_train / len(train_loader)}")
    return


# useful function to show an image that takes a tensor as an input
def show_img(t_img:torch.tensor) -> None:
    if len(t_img.shape > 3):
        assert t_img.shape[0] <= 1, "You're trying to display a batch of size > 1. Be wary of opening more than one image at a time..."
    to_img = transforms.ToPILImage()
    img = to_img(t_img)
    img.show()
    return

#################################### MODEL ####################################


class MyModel(nn.Module):
    def __init__(self, n_chans1=32):
        super(MyModel, self).__init__()

        # data preprocessing
        self.gaussian_blur = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.edge_detection = nn.Conv2d(3, 3, kernel_size=3, padding=1)

        self.n_chans1 = n_chans1

        # first convolution and batch normalization
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1)

        # second convolution and batch normalization
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_chans1 // 2)

        # linear layers
        self.fc1 = nn.Linear(2 * 8 * 8 * n_chans1, 32)
        self.fc2 = nn.Linear(32, 8)

        with torch.no_grad():
            self.gaussian_blur.bias.zero_()
            self.edge_detection.bias.zero_()
            self.gaussian_blur.weight[:] = kernels.gaussian_blur3x3
            self.edge_detection.weight[:] = kernels.edge_detection_RIDGE0

    # forward pass
    def forward(self, x):
        out = self.gaussian_blur(x)
        out = self.edge_detection(out)
        out = self.conv1_batchnorm(self.conv1(x))
        out = F.max_pool2d(torch.tanh(out), 2)
        out = self.conv2_batchnorm(self.conv2(out))
        out = F.max_pool2d(torch.tanh(out), 2)
        out = out.view(-1, 2 * 8 * 8 * self.n_chans1)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

####################################### GLOBAL VARIABLES ##########################################
path = "~/"
assert (path != ""), "[*]ERROR: You forgot to include the data path in the path variable"
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')) # train on gpu if available
model_path = "~/"
assert (model_path != ""), "[*]ERROR: You forgot to include the model path in the model_path variable"
model_name = "xraymodel.pt"
assert (model_name != ""), "[*]Error: You forgot to include the model name in the model_name variable"


################################## SCRIPT #######################################
# resize and transform images into tensors
# normalization is done later, so we don't include it in the first transforms
transformations = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

print(f"[*] LOADING IMAGES")
datafolder = torchvision.datasets.ImageFolder(root=path, transform=transformations)
datafolder.classes = [(x, c) for c, x in enumerate(datafolder.classes)]
print(f"Classes: {datafolder.classes}")

# split the data into training and validation data
train_data, val_data = train_val_split(datafolder)

# create the training and validation tensors
print(f"[*] Creating training and validation tensors")
train_imgs = {c:[] for c, x in enumerate(datafolder.classes)} # train_imgs.keys() are integers representing classes
val_imgs = {c:[] for c, x in enumerate(datafolder.classes)}


# extract images and labels from the DataLoader
print(f"[*] Extracting images and labels from the DataLoader")
for c, (img_t, label) in enumerate(train_data):
    if c % 1000 == 0:
        print(f"Processing img {c}")
    train_imgs[label].append(img_t)

for c, (img_t, label) in enumerate(val_data):
    if c % 1000 == 0:
        print(f"Processing img {c}")
    val_imgs[label].append(img_t)

# stack the images into an extra dimension
print(f"stacking the images into an extra dimension")
for key in train_imgs:
    train_imgs[key] = torch.stack(train_imgs[key])

for key in val_imgs:
    val_imgs[key] = torch.stack(val_imgs[key])

# normalize the images belonging to each class!
# NOTE: normalize for each class,
# not over all the images

ntrain_imgs = []
nval_imgs = []

# normalize the training set
print(f"[*] Normalizing the training set")
class_means = {}
class_std = {}

for key in train_imgs:
    class_means[key] = torch.mean(train_imgs[key], (0, 2, 3))
    class_std[key] = torch.std(train_imgs[key], (0, 2, 3))
    ntrain_imgs.append(((train_imgs[key] - class_means[key][None, :, None, None]) / class_std[key][None, :, None, None], key))

# normalize the validation set
print(f"[*] Normalizing the validation set")
class_means = {}
class_std = {}

for key in val_imgs:
    class_means[key] = torch.mean(val_imgs[key], (0, 2, 3))
    class_std[key] = torch.std(val_imgs[key], (0, 2, 3))
    #class_std[key] = torch.std(train_imgs[key], (0, 2, 3)) # validation performance is slightly better if we use the standard deviation of the training set... why?
    nval_imgs.append(((val_imgs[key] - class_means[key][None, :, None, None]) / class_std[key][None, :, None, None], key))

# further preprocess necessary to perfom forward pass
print(f"[*] Preprocessing for forward pass")
n_train_imgs = []
for label in range(len(ntrain_imgs)):
    for k in range(ntrain_imgs[label][0].shape[0]):
        n_train_imgs.append((ntrain_imgs[label][0][k], label))
print(len(n_train_imgs))

n_val_imgs = []
for label in range(len(nval_imgs)):
    for k in range(nval_imgs[label][0].shape[0]):
        n_val_imgs.append((nval_imgs[label][0][k], label))
print(len(n_val_imgs))

# create loaders for training set and validation set.
print(f"[*] Create loaders for training set and validation set")
train_loader = torch.utils.data.DataLoader(n_train_imgs, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(n_val_imgs, batch_size=64, shuffle=True)

for img_t, label in train_loader:
    print(img_t.shape, label, type(label), len(label), sep="\n")
    break
for imt_t, label in val_loader:
    print(img_t.shape, label, type(label), len(label), sep="\n")
    break

n_out = len(datafolder.classes)
print(f"[*] Dataset contains {n_out} classes)")

model = MyModel()


# load previously trained state dict if it exists
if os.path.exists(model_path + model_name):
    print(f"[*] Resuming training. Loading previous state dict")
    model.load_state_dict(torch.load(model_path + model_name))

numel_list = [p.numel() for p in model.parameters()]
print("[*] Number of parameters:", sum(numel_list), numel_list)


optimizer = optim.SGD(model.parameters(), lr=.6e-2, weight_decay=1e-3) # NOTE: weight_decay acts like l2 regularization
loss_fn = nn.CrossEntropyLoss()
n_epochs = 60

# train the model
print(f"[*] TRAINING for {n_epochs} epochs")
training_loop(
     n_epochs = n_epochs,
     optimizer = optimizer,
     model = model,
     loss_fn = loss_fn,
     train_loader = train_loader,
    )

# validation
validate(model, val_loader)

# update the model's state dictionary
torch.save(model.state_dict(), model_path + model_name)
