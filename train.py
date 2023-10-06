import torch
from model import PoseLoss
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model import PoseNet, PoseLoss
from Dataset import DataSource

# TensorBoard 로그 디렉토리 설정
log_dir = "logs"  # 원하는 디렉토리로 변경 가능
writer = SummaryWriter(log_dir)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.0001
batch_size = 75
data_root = './data/KingsCollege/'
EPOCH = 75000
save_root_path="./save_model/"

train_data = DataSource(data_root, train=True)
test_data = DataSource(data_root, train=False)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

posenet = PoseNet.posenet_v1().to(device)

criterion = PoseLoss(0.3, 150, 0.3, 150, 1, 500)
optimizer = torch.optim.Adam(nn.ParameterList(posenet.parameters()), lr=learning_rate)
Best_Loss = 1000000


for epoch in tqdm(range(EPOCH)):
    test_losses_list = []
    train_losses_list = []
    for step, (images, poses) in enumerate(train_loader):
        b_images = Variable(images, requires_grad=True).to(device)
        poses[0] = np.array(poses[0])
        poses[1] = np.array(poses[1])
        poses[2] = np.array(poses[2])
        poses[3] = np.array(poses[3])
        poses[4] = np.array(poses[4])
        poses[5] = np.array(poses[4])
        poses[6] = np.array(poses[5])
        poses = np.transpose(poses)
        b_poses = Variable(torch.Tensor(poses), requires_grad=True).to(device)

        p1_x, p1_q, p2_x, p2_q, p3_x, p3_q = posenet(b_images)
        loss = criterion(p1_x, p1_q, p2_x, p2_q, p3_x, p3_q, b_poses)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_np = loss.detach().numpy()
        train_losses_list.append(train_loss_np)

    
    
    train_losses_np = np.array(train_losses_list)
    mean_train_loss = np.mean(train_losses_np)
    writer.add_scalar("Train Loss", mean_train_loss, epoch)  
    print("#" * 50)
    print(f"Train Epoch {epoch} Loss : {mean_train_loss}")
    print("*"*50)

    test_losses_list = []
    for step, (images, poses) in enumerate(test_loader):
        with torch.no_grad():
            # b_images = Variable(images, requires_grad=True).to(device)
            b_images = images.detach().to(device)

            poses[0] = np.array(poses[0])
            poses[1] = np.array(poses[1])
            poses[2] = np.array(poses[2])
            poses[3] = np.array(poses[3])
            poses[4] = np.array(poses[4])
            poses[5] = np.array(poses[4])
            poses[6] = np.array(poses[5])
            poses = np.transpose(poses)
            b_poses = Variable(torch.Tensor(poses), requires_grad=False).to(device)

            p1_x, p1_q, p2_x, p2_q, p3_x, p3_q = posenet(b_images)
            loss = criterion(p1_x, p1_q, p2_x, p2_q, p3_x, p3_q, b_poses)
            loss_np = loss.numpy()
            test_losses_list.append(loss_np)
            
    test_losses_np = np.array(test_losses_list)
    mean_test_loss = np.mean(test_losses_np)
    writer.add_scalar("Test Loss", mean_test_loss, epoch)    

    print("#" * 50)
    print(f"Test Epoch {epoch} Loss : {mean_test_loss}")
    print("*"*50)

    if Best_Loss > mean_test_loss:
        Best_Loss = mean_test_loss
        print("#" * 50)
        print("New Best Model Pops up")
        print(f"Test Mean Loss : {Best_Loss}")

        save_path = save_root_path + "posenet_20231006_epoch_" + str(epoch) + ".pth"
        torch.save(posenet.state_dict(), save_path)
writer.close()
