import torch
import torch.nn as nn

def train_dcgan(generator, discriminator, train_loader, beta1, epochs, lr, device, criterion, optimizerD, optimizerG):
    for epoch in range(epochs):
        for i, (real_images, labels) in enumerate(train_loader):
            batch_size = real_images.size(0)
            fake_labels = torch.zeros(batch_size).to(device)
            real_labels = torch.ones(batch_size).to(device)

            # Huấn luyện Discriminator
            discriminator.zero_grad()
            real_images = real_images.to(device)
            labels = labels.to(device).float()
            outputs = discriminator(real_images).view(-1)
            d_loss_real = criterion(outputs, labels)
            d_loss_real.backward()
            optimizerD.step()

            z = torch.randn(batch_size, 100, 1, 1).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach()).view(-1)
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss_fake.backward()
            d_loss = d_loss_real + d_loss_fake
            optimizerD.step()

            generator.zero_grad()
            output = discriminator(fake_images).view(-1)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            optimizerG.step()

            print(f"Epoch [{epoch}/{epochs}] Batch {i}/{len(train_loader)} Loss D: {d_loss.item()} Loss G: {g_loss.item()}")

    return generator, discriminator
