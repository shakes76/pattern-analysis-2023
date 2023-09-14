'''
InfoVAE test script for MNIST
'''
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

#parameters
num_epochs = 24
depth = 64
latent_size = 2
in_channels = 1
batch_size = 64
num_to_generate = 16

#network IO paths
model_name = "InfoVAE-2D"
path = '/home/uqscha22/'
# path = 'F:/'
output_path = path+"data/mnist/output_torch_"+model_name+"/"

if not os.path.isdir(output_path):
    os.makedirs(output_path)

#--------------
#Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

trainset = torchvision.datasets.MNIST(root=path+'data/mnist', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True) #num_workers=6
total_step = len(train_loader)
# print(total_step)

#--------------
# Model
class ConvBlock(nn.Module):
    '''
    Basic Conv Block
    '''
    def __init__(self, in_planes, planes, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=kernel_size, stride=stride, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        return F.relu(self.bn1(self.conv1(x)))

class ConvTransposeBlock(nn.Module):
    '''
    Basic Conv Transpose Block
    '''
    def __init__(self, in_planes, planes, kernel_size=3, stride=1):
        super(ConvTransposeBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_planes, planes, kernel_size=kernel_size, stride=stride, bias=False, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        return F.relu(self.bn1(self.conv1(x)))

# Encoder, Decder
class Encoder(nn.Module):
    '''
    VAE Encoder
    '''
    def __init__(self, z_dim, block, depth, in_channels, kernel_size=3):
        super().__init__()
        self.z_dim = z_dim
        self.block = block
        self.depth = depth
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self._make_layers(self.z_dim, self.block, self.depth, self.in_channels, self.kernel_size)

    def _make_layers(self,z_dim, block, depth, in_channels, kernel_size):
        ''' Build the layers as static PyTorch attributes '''
        self.block_conv1 = block(in_channels, depth, kernel_size=kernel_size, stride=2)
        self.block_conv2 = block(depth, depth*2, kernel_size=kernel_size, stride=2)
        self.linear1 = nn.Linear(7*7*depth*2, 128)
        self.linear2 = nn.Linear(128, 32)
        self.linear3 = nn.Linear(32, z_dim)

    def forward(self, x):
        out = self.block_conv1(x)
        out = self.block_conv2(out)
        # print(out.shape)
        out = out.view(out.size(0), -1) #flatten
        # print(out.shape)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        return out
    
class Decoder(nn.Module):
    '''
    VAE Decoder
    '''
    def __init__(self, z_dim, block, depth, in_channels, kernel_size=3):
        super().__init__()
        self.z_dim = z_dim
        self.block = block
        self.depth = depth
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self._make_layers(self.z_dim, self.block, self.depth, self.in_channels, self.kernel_size)

    def _make_layers(self,z_dim, block, depth, in_channels, kernel_size):
        ''' Build the layers as static PyTorch attributes '''
        self.linear1 = nn.Linear(z_dim, 32)
        self.linear2 = nn.Linear(32, 128)
        self.linear3 = nn.Linear(128, 7*7*depth*2)
        self.block_conv1 = block(depth*2, depth, kernel_size=kernel_size, stride=2)
        self.block_conv2 = block(depth, in_channels, kernel_size=kernel_size, stride=2)

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.linear3(out)
        # print(out.shape)
        out = out.view(-1, self.depth*2, 7, 7) #reshape
        # print(out.shape)
        out = self.block_conv1(out)
        out = self.block_conv2(out)
        # print(out.shape)
        return out
    
class VAE(nn.Module):
    '''
    VAE
    '''
    def __init__(self, z_dim, depth, in_channels, kernel_size=3):
        super().__init__()
        self.encoder = Encoder(z_dim, ConvBlock, depth, in_channels, kernel_size)
        self.decoder = Decoder(z_dim, ConvTransposeBlock, depth, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon
    
def encoder_loss(latent):
    '''
    Compute MMD loss for the InfoVAE
    '''
    def compute_kernel(x, y):
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]
        tiled_x = torch.tile(torch.reshape(x, [x_size, 1, dim]), [1, y_size, 1]).to(device)
        tiled_y = torch.tile(torch.reshape(y, [1, y_size, dim]), [x_size, 1, 1]).to(device)
        return torch.exp(-torch.mean(torch.square(tiled_x - tiled_y), axis=2) / float(dim))

    def compute_mmd(x, y):
        x_kernel = compute_kernel(x, x)
        y_kernel = compute_kernel(y, y)
        xy_kernel = compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    'So, we first get the mmd loss'
    'First, sample from random noise'
    batch_size = latent.shape[0]
    latent_dim = latent.shape[1]
    true_samples = torch.randn((batch_size, latent_dim))
    'calculate mmd loss'
    loss_mmd = compute_mmd(true_samples, latent)

    'Add them together, then you can get the final loss'
    return loss_mmd

model = VAE(latent_size, depth, in_channels)
model = model.to(device)

#model info
print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
print(model)

criterion_enc = encoder_loss
criterion_dec = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#--------------
# Train the model
def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    with torch.no_grad():
        predictions = model(test_input)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, 0, :, :].cpu().detach().numpy(), cmap='gray')
        plt.axis('off')

    plt.savefig(output_path+'image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

seed = torch.randn([num_to_generate, latent_size]).to(device)

model.train()
print("> Training")
start = time.time() #time generation
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): #load a batch
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        latent_codes, recons = model(images)
        #losses
        mmd_loss = criterion_enc(latent_codes)
        recon_loss = criterion_dec(images, recons)
        loss = mmd_loss + recon_loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 900 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            # generate_and_save_images(model.decoder, epoch, seed)
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

generate_and_save_images(model.decoder, epoch, seed)
