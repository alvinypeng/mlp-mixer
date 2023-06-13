import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms

from architecture import MLPMixer


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def train(model: nn.Module, device, train_loader, optimizer, epoch) -> None:
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
            
def test(model: nn.Module, device, test_loader, epoch, writer):
    model.eval()
    
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))
        
    writer.add_scalar('Test loss', test_loss, epoch)
    writer.add_scalar('Test accuracy', test_accuracy, epoch)
    

def main():
    parser = argparse.ArgumentParser(description='MLP-Mixer for Food101')
    
    # Paths
    parser.add_argument('--data_path', type=str, default='Food101',
                        help='path to dataset')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pt',
                        help='path to save/load checkpoint')
    parser.add_argument('--tensorboard_path', type=str, default='log',
                        help='path for tensorboard paths')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of worker threads for data loading')
        
    # Training hyperparameters
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='number of samples in a batch')
    parser.add_argument('--lr', default=3e-4, type=float,
                        help='learning rate for ADAM optimizer')
    
    args = parser.parse_args()
    
    # Set up device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Appropriate transforms (https://github.com/Prakhar998/food-101)
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30), 
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])    
    
    # Set up datasets
    train_dataset = torchvision.datasets.Food101(root=args.data_path, 
                                                 download=True,
                                                 transform=train_transforms)
    test_dataset = torchvision.datasets.Food101(root=args.data_path, 
                                                split='test', 
                                                transform=test_transforms)
    
    # Set up loaders
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   num_workers=args.num_workers, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    # Set up model (S/32 model as described in original paper)
    model = MLPMixer(image_dim=(3, 224, 224), patch_size=32, token_dim=512, 
                     token_mixing_dim=256, channel_mixing_dim=2048, 
                     n_blocks=8, n_classes=101).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Load checkpoint if specified
    if os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        logger.info(f'Loaded checkpoint from epoch {epoch}')
    else:
        epoch = 0

    # For tensorboard graphs
    writer = SummaryWriter(args.tensorboard_path)
        
    # Train and validate for specified number of epochs
    for epoch in range(epoch, args.epochs):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch, writer)
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, args.checkpoint_path)
        logger.info(f'Saved checkpoint at epoch {epoch}')

    
if __name__ == '__main__':
    main()

    