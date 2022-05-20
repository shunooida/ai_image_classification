import torch

import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL.Image
import torchvision

def cam_for_resnet(model, img_fpath):

    img = PIL.Image.open(img_fpath).convert('RGB')
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = transforms(img)
    img = img.unsqueeze(0)

    #モデルを推論モードにする
    model.eval()

    #最後の畳み込み層の特徴マップを取得
    features = model.conv1(img)
    features = model.bn1(features)
    features = model.relu(features)
    features = model.maxpool(features)
    features = model.layer1(features)
    features = model.layer2(features)
    features = model.layer3(features)
    features = model.layer4(features)

    #最終的な特徴マップ
    final_features = features

    features = model.avgpool(features)
    features = features.view(features.size(0), -1)
    #最終の出力
    output = model.fc(features)
    tensor_max_index = torch.argmax(output)
    #正解クラスのインデックス
    max_index = int(tensor_max_index)
    #分類クラスの数
    number_of_class = torch.numel(output)

    param_list = []

    for name, param in model.state_dict().items():
        param_list.append(param)

    weight_fc = param_list[-2]
    #特徴マップに掛け合わせる
    feature_weights = weight_fc[max_index]

    #重みと特徴マップを掛け合わせて総和を求める
    final_map = 0
    for i in range(number_of_class):
        #feature_weights[i]に関してはエラーにならない
        #print(final_features.shape)
        map = final_features[0][i] * feature_weights[i]
        final_map += map

    #画像をプロットするための値の正規化
    heatmap = final_map / torch.max(final_map)
    heatmap = heatmap.detach().numpy()

    #元の画像にヒートマップを重ねる
    img = cv2.imread(img_fpath)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.uint8(255 * superimposed_img / np.max(superimposed_img))
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    plt.imshow(superimposed_img)
    plt.show()

    