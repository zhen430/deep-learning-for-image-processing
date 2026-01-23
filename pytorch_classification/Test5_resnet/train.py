import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34

import wandb

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    print(os.getcwd())
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())  #键值对互换
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,       #批处理
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu', weights_only = False))
    # for param in net.parameters():   #只训练分类头
    #     param.requires_grad = False

    # change fc layer structure，改变最后分类头的类别数（原本默认1000类别）
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    lr = 0.0001
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)

    epochs = 3
    best_acc = 0.0
    save_path = './resNet34.pth'
    train_steps = len(train_loader)

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="liu2000-kth",
        # Set the wandb project where this run will be logged.
        project="prac_202601_wb",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": lr,
            "architecture": "Resnet34",
            "dataset": "Flower-5",
            "epochs": epochs,
        },
        notes = '不再把wandb.init写在前面，让config参数跟随系统。把wandb.log分开写在不同位置，记录训练的2个loss、验证的acc和loss。',
    )

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0   #记录一个epoch的所有step的loss。但其实最后打印的/呈现的应该是平均loss所以还要除以step数。
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()   #把上一步计算出的并暂存的梯度清空
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            #logits是模型最后一层整形到5维的输出，但没有经过softmax映射成概率。nn.CrossEntropyLoss内部会先做softmax。
            #label就是类别数字，是标量，nn.CrossEntropyLoss内部会先做one-hot。
            loss.backward()   #求导，计算梯度并暂存
            optimizer.step()   #更新梯度

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

            run.log({"train/loss_per_step": loss})
        run.log({"train/loss": running_loss / train_steps})

        # validate
        net.eval()  #关闭dropout层，让BN层用训练时候的总体均值方差、而不是这个推理batch的均值方差
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            val_loss = 0
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                val_loss += loss_function(outputs, val_labels.to(device))  #通常验证机只关心准确率，不关心loss，也可以注释掉。
                predict_y = torch.max(outputs, dim=1)[1]   #类似于softmax，但是为了简化计算没有用softmax而是直接max了。
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()  #torch.eq是做比较，看有几个相等的

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num  #val_num是整个验证集大小，val_accurate是针对单一样本说的所以除以它
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)  #所有权重，不包括模型结构


        run.log({"val/acc": val_accurate, "val/loss": val_loss/len(validate_loader)})

    run.finish()
    print('Finished Training')


if __name__ == '__main__':
    main()
