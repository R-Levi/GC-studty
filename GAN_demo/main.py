import  argparse
import torch
import os
from torchvision import datasets,transforms
from torchvision.utils import save_image
from GAN_demo.Discriminator import *
from GAN_demo.Generator import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--numworks", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--hidden_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

if __name__ == '__main__':
    image_shape = (opt.channels,opt.img_size,opt.img_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #TODO 模型
    generator = Generator(opt,image_shape)
    discriminator = Discriminator(opt,image_shape)

    # TODO 准备数据
    os.makedirs("data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )
    #TODO 优化器 loss
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    adv_loss = torch.nn.BCELoss()

    generator.to(device)
    discriminator.to(device)
    adv_loss.to(device)
    #TODO train
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            #判别器验证 GT
            valid = torch.ones((imgs.shape[0], 1),requires_grad=False,dtype=torch.float32).to(device)
            fake = torch.zeros((imgs.shape[0], 1), requires_grad=False,dtype=torch.float32).to(device)

            real_imgs = imgs.to(device)

            #TODO Train Generator
            optimizer_G.zero_grad()
            # 输入噪音图片
            z = torch.tensor(np.random.normal(0,1,(imgs.shape[0],opt.hidden_dim)),requires_grad=True,dtype=torch.float32).to(device)
            # 生成图片
            gen_imgs = generator(z)
            g_loss = adv_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            #TODO Train Discriminator
            optimizer_D.zero_grad()

            real_loss = adv_loss(discriminator(real_imgs), valid)
            fake_loss = adv_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

    ## 保存模型
    torch.save(generator.state_dict(), 'save/gan/generator.pth')
    torch.save(discriminator.state_dict(), 'save/gan/discriminator.pth')

    # generator = Generator()
    # discriminator = Discriminator()
    # generator.load_state_dict(torch.load('./save/gan/discriminator.pth'))
    # discriminator.eval()
