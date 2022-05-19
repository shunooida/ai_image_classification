import torch

import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL.Image
import torchvision
import torch.nn.functional as F

def cam_for_multi(model, img_fpath):

    img = PIL.Image.open(img_fpath).convert('RGB')
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = transforms(img)
    img = img.unsqueeze(0)

    model.eval()#モデルを推論モードにする

    #最後の畳み込み層の特徴マップを取得
    conv_input = model.conv_model(img)

    first_input = model.conv1(img)
    first_input = model.maxpool1(first_input)

    second_input = model.conv2(img)
    second_input = model.maxpool2(second_input)

    features = torch.cat([conv_input, first_input, second_input], 1)

    #最終的な特徴マップ
    final_features = second_input

    merge_gap = F.adaptive_avg_pool2d(features, (1, 1))
    merge = merge_gap.view(merge_gap.size(0), -1)

    #最終の出力
    output = model.fc(merge)

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

    feature_weights_3 = feature_weights[1024:]

    #重みと特徴マップを掛け合わせて総和を求める
    final_map = 0
    for i in range(number_of_class):
        map = final_features[0][i] * feature_weights_3[i]
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

    