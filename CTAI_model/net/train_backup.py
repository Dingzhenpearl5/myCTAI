import sys

sys.path.append("..")
import torch
from torch.nn import init
from torch.utils.data import DataLoader
from data_set import make
from net import unet
from utils import dice_loss
import matplotlib.pyplot as plt
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
res = {'epoch': [], 'loss': [], 'dice': []}


def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv3d') != -1:
        init.xavier_normal(m.weight.data, 0.0)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, 0.0)
        init.constant_(m.bias.data, 0.0)


# 参数
rate = 0.50
learn_rate = 0.001
epochs = 50  # 增加训练轮数以提高模型性能
# train_dataset_path = '../data/all/d1/'
train_dataset_path = 'C:/Users/Masoa/OneDrive/work/CTAI/src/train'  # 修复：使用正确的数据路径

train_dataset, test_dataset = make.get_d1(train_dataset_path)
unet = unet.Unet(1, 1).to(device).apply(weights_init)
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.Adam(unet.parameters(), learn_rate)


def train():
    global res
    dataloaders = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    for epoch in range(epochs):
        dt_size = len(dataloaders.dataset)
        epoch_loss, epoch_dice = 0, 0
        step = 0
        for x, y in dataloaders:
            id = x[1:]
            step += 1
            x = x[0].to(device)
            y = y[1].unsqueeze(1).to(device)  # 添加 channel 维度 [B, H, W] -> [B, 1, H, W]
            # print(x.size())
            # print(y.size())
            optimizer.zero_grad()
            outputs = unet(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            # 计算 dice 系数
            a = outputs.cpu().detach().squeeze(1).numpy()
            
            # 调试：查看模型输出范围
            if step % 100 == 0:
                print(f"\n[DEBUG] Output range: min={a.min():.4f}, max={a.max():.4f}, mean={a.mean():.4f}")
                print(f"[DEBUG] Mask sum: {b.sum() if 'b' in locals() else 'N/A'}")
            
            a[a >= rate] = 1
            a[a < rate] = 0
            b = y.cpu().detach().squeeze(1).numpy()
            dice = dice_loss.dice(a, b)
            epoch_loss += float(loss.item())
            epoch_dice += dice

            if step % 100 == 0:
                res['epoch'].append((epoch + 1) * step)
                res['loss'].append(loss.item())
                print("Epoch%d Step%d/%d | Loss:%.4f | Train Dice:%.4f | " % (
                    epoch + 1, step, (dt_size - 1) // dataloaders.batch_size + 1, 
                    loss.item(), dice), end='')
                test()
        
        # 每个 epoch 结束后打印统计
        avg_loss = epoch_loss / step
        avg_dice = epoch_dice / step
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs} Complete | Avg Loss: {avg_loss:.4f} | Avg Dice: {avg_dice:.4f}")
        print(f"{'='*60}\n")
    #  print("epoch %d loss:%0.3f,dice %f" % (epoch, epoch_loss / step, epoch_dice / step))
    
    # 保存模型
    torch.save(unet.state_dict(), '../model_weights.pth')
    print(f"\nModel saved to model_weights.pth")
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(res['epoch'], np.squeeze(res['loss']), label='Train loss')
    plt.ylabel('Loss')
    plt.xlabel('Steps')
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(res['epoch'], np.squeeze(res['dice']), label='Test Dice', color='#FF9966')
    plt.ylabel('Dice Score')
    plt.xlabel('Steps')
    plt.title("Validation Dice Score")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_history.jpg")
    print(f"Training curves saved to training_history.jpg")

    # torch.save(unet, 'unet.pkl')
    # model = torch.load('unet.pkl')
    test()


def test():
    global res, img_y, mask_arrary
    epoch_dice = 0
    with torch.no_grad():
        dataloaders = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
        for x, mask in dataloaders:
            id = x[1:]  # ('1026',), ('10018',)]先病人号后片号
            x = x[0].to(device)
            y = unet(x)
            mask_arrary = mask[1].cpu().squeeze(0).detach().numpy()
            img_y = torch.squeeze(y).cpu().numpy()
            img_y[img_y >= rate] = 1
            img_y[img_y < rate] = 0
            img_y = img_y * 255
            epoch_dice += dice_loss.dice(img_y, mask_arrary)
            # cv.imwrite(f'data/out/{mask[0][0]}-result.png', img_y, (cv.IMWRITE_PNG_COMPRESSION, 0))
        print('test dice %f' % (epoch_dice / len(dataloaders)))
        res['dice'].append(epoch_dice / len(dataloaders))


if __name__ == '__main__':
    train()
    test()
