import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
import model_utils
import net
import resnet

resnet_transforms = {
            'train': transforms.Compose([
                transforms.Scale((224,224)),
                # transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                # transforms.RandomSizedCrop(448),
                transforms.Scale((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                # transforms.RandomSizedCrop(448),
                transforms.Scale(224,224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
}


def generate_dataloader(path=".",batch_size=32,transforms=resnet_transforms):
    three_set_dict={}
    train_set={}
    train_path=os.path.join(path,"train")
    for i in range(0,65):
        current_path=os.path.join(train_path,str(i))
        for name in os.listdir(current_path):
            if(".jpg" in name):
                train_set[str(i)+"/"+name[:-4]]=[i,name[:-4]]

    validate_set={}
    validate_path=os.path.join(path,"valid")
    for i in range(0,65):
        current_path=os.path.join(validate_path,str(i))
        for name in os.listdir(current_path):
            if(".jpg" in name):
                validate_set[str(i)+"/"+name[:-4]]=[i,name[:-4]]

    three_set_dict["train"]=train_set
    three_set_dict["val"]=validate_set
    three_set_dict["test"]=validate_set
    train_loader=Data.DataLoader(model_utils.common_dataset.three_set_dataset(three_set_dict,os.path.join(path,"train"),"train",transforms),batch_size=batch_size,shuffle=True,num_workers=4)
    val_loader=Data.DataLoader(model_utils.common_dataset.three_set_dataset(three_set_dict,os.path.join(path,"valid"),"val",transforms),batch_size=10,shuffle=True,num_workers=4)
    test_loader=Data.DataLoader(model_utils.common_dataset.three_set_dataset(three_set_dict,os.path.join(path,"valid"),"val",transforms),batch_size=10,shuffle=True,num_workers=4)
    return train_loader,val_loader,test_loader

def optimizers_generator(models,lr_base,lr_fc,weight_decay=0.0005,paral=False):
    optimizers=[]
    model=models[0]
    c_param=None
    if(paral):
        c_param=model.module.fc.parameters()
    else:
        c_param=model.fc.parameters()

    clssify_params = list(map(id, c_param))
    base_params = filter(lambda p: id(p) not in clssify_params,model.parameters())
    optimizer = torch.optim.SGD([
        {'params': base_params,'lr':lr_base},
        {'params': c_param, 'lr': lr_fc},
        ], lr_base, momentum=0.9, weight_decay=weight_decay)
    optimizers.append(optimizer)
    return optimizers

configA=model_utils.config.config()
configA["epochs"]=100
configA["learning_rate_decay_epochs"]=[50,75]
configA["dataset_function"]=generate_dataloader
configA["dataset_function_params"]={"batch_size":32}

configB=model_utils.config.config()
configB["epochs"]=200
configB["learning_rate_decay_epochs"]=[120,160]
configB["dataset_function"]=generate_dataloader
configB["dataset_function_params"]={"batch_size":32}

configC=model_utils.config.config()
configC["epochs"]=300
configC["learning_rate_decay_epochs"]=[180,240]
configC["dataset_function"]=generate_dataloader
configC["dataset_function_params"]={"batch_size":32}

config_tsne=model_utils.config.config()
config_tsne["restored_path"]="./checkpoints/task_C/201904151010/best/"
config_tsne["dataset_function"]=generate_dataloader
config_tsne["dataset_function_params"]={"batch_size":32}

r=model_utils.runner.runner()

tasks=[]
t_A={
"task_name":"task_A",
"solver":{"class":model_utils.solver.common_classify_solver,"params":{}},
"models":[{"class":net.resnet50,"params":{"pretrain":True}}],
"optimizers":{"function":optimizers_generator,"params":{"lr_base":0.001,"lr_fc":0.01,"weight_decay":0.0005,"paral":False}},
"config":configA,
"mem_use":[3000]
}

t_B={
"task_name":"task_B",
"solver":{"class":model_utils.solver.common_classify_solver,"params":{}},
"models":[{"class":net.resnet50,"params":{"pretrain":False}}],
"optimizers":{"function":optimizers_generator,"params":{"lr_base":0.1,"lr_fc":0.1,"weight_decay":0.0005,"paral":False}},
"config":configB,
"mem_use":[3000]
}

t_C={
"task_name":"task_C",
"solver":{"class":model_utils.solver.common_classify_solver,"params":{}},
"models":[{"class":resnet.ResNet,"params":{}}],
"optimizers":{"function":optimizers_generator,"params":{"lr_base":0.1,"lr_fc":0.1,"weight_decay":0.0005,"paral":False}},
"config":configC,
"mem_use":[3000]
}

t_sne={
"task_name":"t_SNE",
"solver":{"class":model_utils.solver.t_sne_solver,"params":{}},
"models":[{"class":net.my_net,"params":{"type":"t_SNE"}}],
"optimizers":{"function":optimizers_generator,"params":{"lr_base":0.01,"lr_fc":0.01,"weight_decay":0.0005,"paral":False}},
"config":config_tsne,
"mem_use":[3000]
}



# tasks.append(t_A)
# tasks.append(t_B)
tasks.append(t_C)
# tasks.append(t_sne)
# print(resnet.ResNet())

r.generate_tasks(tasks)
r.main_loop()
