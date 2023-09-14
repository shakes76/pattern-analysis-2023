'''
Generative Adversarial Networks on the MNIST dataset
'''
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

#parameters
num_epochs = 40
depth = 32
latent_size = 7
in_channels = 1
kernel = 3
batch_size = 100
num_to_generate = 16

#network IO paths
model_name = "GAN"
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
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x)))

class ConvTransposeBlock(nn.Module):
    '''
    Basic Conv Transpose Block
    '''
    def __init__(self, in_planes, planes, kernel_size=3, stride=1):
        super(ConvTransposeBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_planes, planes, kernel_size=kernel_size, stride=stride, bias=False, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x)))

class Generator(nn.Module):
    '''
    GAN Generator G
    '''
    def __init__(self, z_dim, block, upblock, depth, in_channels, kernel_size=3):
        super().__init__()
        self.z_dim = z_dim
        self.block = block
        self.upblock = upblock
        self.depth = depth
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self._make_layers(self.z_dim, self.block, self.upblock, self.depth, self.in_channels, self.kernel_size)

    def _make_layers(self, z_dim, block, upblock, depth, in_channels, kernel_size):
        ''' Build the layers as static PyTorch attributes '''
        self.block_conv1a = block(1, depth*4, kernel_size=kernel_size, stride=1)
        self.block_conv1b = block(depth*4, depth*4, kernel_size=kernel_size, stride=1)
        self.block_conv1c = upblock(depth*4, depth*2, kernel_size=kernel_size, stride=2)
        self.block_conv2a = block(depth*2, depth*2, kernel_size=kernel_size, stride=1)
        self.block_conv2b = block(depth*2, depth, kernel_size=kernel_size, stride=1)
        self.block_conv2c = upblock(depth, depth, kernel_size=kernel_size, stride=2)
        self.conv1 = nn.Conv2d(depth, in_channels, kernel_size, stride=1, bias=False, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # print(out.shape)
        out = self.block_conv1a(x)
        out = self.block_conv1b(out)
        out = self.block_conv1c(out)
        # print(out.shape)
        out = self.block_conv2a(out)
        out = self.block_conv2b(out)
        out = self.block_conv2c(out)
        out = self.conv1(out)
        out = self.tanh(out)
        # print(out.shape)
        return out

class Discriminator(nn.Module):
    '''
    GAN Discriminator D
    '''
    def __init__(self, z_dim, block, depth, in_channels, kernel_size=3):
        super().__init__()
        self.z_dim = z_dim
        self.block = block
        self.depth = depth
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self._make_layers(self.z_dim, self.block, self.depth, self.in_channels, self.kernel_size)

    def _make_layers(self, z_dim, block, depth, in_channels, kernel_size):
        ''' Build the layers as static PyTorch attributes '''
        self.block_conv1 = block(in_channels, depth, kernel_size=kernel_size, stride=2)
        self.block_conv2 = block(depth, depth*2, kernel_size=kernel_size, stride=2)
        self.linear1 = nn.Linear(7*7*depth*2, 64)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.linear2 = nn.Linear(64, z_dim)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.block_conv1(x)
        out = self.block_conv2(out)
        # print(out.shape)
        out = out.view(out.size(0), -1) #flatten
        # print(out.shape)
        out = self.relu1(self.linear1(out))
        out = self.relu2(self.linear2(out))
        out = self.sigmoid(out)
        return out

criterion = nn.BCELoss()

#G Network components
G = Generator(latent_size, ConvBlock, ConvTransposeBlock, depth, in_channels, kernel)
G = G.to(device)
#model info
print("Model No. of Parameters:", sum([param.nelement() for param in G.parameters()]))
print(G)

G_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4)

#D Network components
D = Discriminator(1, ConvBlock, depth, in_channels, kernel)
D = D.to(device)
#model info
print("Model No. of Parameters:", sum([param.nelement() for param in D.parameters()]))
print(D)

D_optimizer = torch.optim.Adam(D.parameters(), lr=1e-4)

#--------------
# Train the model
# to visualize progress of same set in the animated GIF
seed = torch.randn(num_to_generate, 1, latent_size, latent_size, device=device)

def generate_and_save_images(model, epoch, test_input):
    # Notice training is set to false via no_grad
    # This is so all layers run in inference mode (batchnorm).
    with torch.no_grad():
        predictions = model(test_input)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, 0, :, :].cpu().numpy(), cmap='gray')
        plt.axis('off')

    plt.savefig(output_path+'image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def D_train(x):
    #Compute loss for real (1) and fake (0)
    D.zero_grad()

    #real images with D
    real_output = D(x)
    real_loss = criterion(real_output, torch.ones_like(real_output, device=device))

    #fake images with D
    z = torch.randn(batch_size, 1, latent_size, latent_size, device=device)
    x_fake = G(z)
    fake_output = D(x_fake)
    fake_loss = criterion(fake_output, torch.zeros_like(fake_output, device=device))

    #backprop parameters
    #optimize for log(D(x)) + log(1-D(G(z)))
    D_loss = real_loss + fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.item()

def G_train():
    # Ideally the loss returns close to ones if generator is successful, 
    # i.e. fake output is good
    G.zero_grad()

    #sample Gaussian distribution
    z = torch.randn(batch_size, 1, latent_size, latent_size, device=device)

    #update G with D(G(z))
    x_fake = G(z)
    fake_output = D(x_fake)
    loss = criterion(fake_output, torch.ones_like(fake_output, device=device))

    #backprop parameters
    #optimize for log(1-D(G(z)))
    loss.backward()
    G_optimizer.step()
    
    return loss.item()

G.train()
D.train()
#Training loop for networks
for epoch in range(1, num_epochs+1):
    start = time.time()

    #lists for keep track of losses throughout training
    D_losses, G_losses = [], []
    #Process each batch
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.to(device)
        #Train each network
        D_losses.append(D_train(x))
        G_losses.append(G_train())

    D_losses = torch.FloatTensor(D_losses)
    G_losses = torch.FloatTensor(G_losses)

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f, Time: %.2f' % (
            (epoch), num_epochs, torch.mean(D_losses), torch.mean(G_losses), time.time()-start))

G.eval()
generate_and_save_images(G, 1, seed)