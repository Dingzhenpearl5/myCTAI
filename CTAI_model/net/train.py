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


# 参数 - 正式训练配置
rate = 0.50
learn_rate = 0.001
epochs = 50  # 正式训练：50个epoch
# train_dataset_path = '../data/all/d1/'
train_dataset_path = 'C:/Users/Masoa/OneDrive/work/CTAI/src/train'  # 修复：使用正确的数据路径

train_dataset, test_dataset = make.get_d1(train_dataset_path)
unet = unet.Unet(1, 1).to(device).apply(weights_init)

# 使用 Dice Loss + BCE Loss 组合来处理类别不平衡
# 肿瘤像素仅占约0.5%,单纯BCE会导致模型倾向于全预测背景
criterion_bce = torch.nn.BCELoss().to(device)

def dice_loss_fn(pred, target, smooth=1.0):
    """Dice Loss for segmentation"""
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return loss.mean()

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
            optimizer.zero_grad()
            outputs = unet(x)
            
            # 组合损失: BCE + Dice Loss (权重比1:1)
            loss_bce = criterion_bce(outputs, y)
            loss_dice = dice_loss_fn(outputs, y)
            loss = loss_bce + loss_dice
            
            loss.backward()
            optimizer.step()
            
            # 计算 dice 系数
            a = outputs.cpu().detach().squeeze(1).numpy()
            a[a >= rate] = 1
            a[a < rate] = 0
            b = y.cpu().detach().squeeze(1).numpy()
            dice = dice_loss.dice(a, b)
            epoch_loss += float(loss.item())
            epoch_dice += dice

            # 每100步输出一次进度
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
            
            # 二值化预测结果
            img_y_binary = img_y.copy()
            img_y_binary[img_y_binary >= rate] = 1
            img_y_binary[img_y_binary < rate] = 0
            
            # 计算 Dice (使用0/1的二值图)
            epoch_dice += dice_loss.dice(img_y_binary, mask_arrary)
            
            # 如需保存可视化结果，使用 0/255
            # img_y_save = img_y_binary * 255
            # cv.imwrite(f'data/out/{mask[0][0]}-result.png', img_y_save, (cv.IMWRITE_PNG_COMPRESSION, 0))
        
        print('test dice %f' % (epoch_dice / len(dataloaders)))
        res['dice'].append(epoch_dice / len(dataloaders))


if __name__ == '__main__':
    train()
    test()
