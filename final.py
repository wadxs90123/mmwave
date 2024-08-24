import torch
import torch.nn as nn
import numpy as np
from KKT_Module.ksoc_global import kgl
from KKT_Module.Configs import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc, ConnectDevice, ResetDevice
from KKT_Module.DataReceive.DataReciever import FeatureMapReceiver
import time

# 載入模型
model_path = r'mmwave_model_final.pth'
device = torch.device('cpu')  

lstm_hidden_size = 128      # LSTM 隱藏層大小 (可調整)
lstm_num_layers = 2         # LSTM 深度 （可調整）

class ConvLSTMNet(nn.Module):
    def __init__(self, lstm_num_layers, lstm_hidden_size, fc_output_size=8):
        super(ConvLSTMNet, self).__init__()
        
        # 定義卷積層
        # RDI & PHD 卷積
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 16x16 feature map
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 8x8 feature map
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 4x4 feature map
        )
         
        # 計算卷積後輸出的特徵圖大小
        # 最終大小為 4x4 , 然後有512個通道
        self.flattened_size = 512 * 4 * 4
        
        # 定義 LSTM 層
        self.lstm = nn.LSTM(
            input_size = self.flattened_size * 2, 
            hidden_size = lstm_hidden_size, 
            num_layers = lstm_num_layers, 
            batch_first = True
        )
        
        # 定義全連接層
        self.fc = nn.Linear(lstm_hidden_size, fc_output_size)
        
    def forward(self, rdi_map, phd_map):
        # 批次大小,  幀數 , 32, 32
        batch_size, timesteps, _, _ = rdi_map.size()
        
        # 初始化 LSTM 隱藏狀態 並確保他的device跟輸入一樣
        h0 = torch.zeros(lstm_num_layers, batch_size, lstm_hidden_size).to(rdi_map.device)
        c0 = torch.zeros(lstm_num_layers, batch_size, lstm_hidden_size).to(rdi_map.device)
        
        # 結合兩個Map的卷積結果
        combined_features = []
        
        for t in range(timesteps):
            rdi_frame = rdi_map[:, t, :, :].unsqueeze(1)  # (batch_size, 1, 32, 32)
            rdi_conv_output = self.conv(rdi_frame)  # (batch_size, 64, 8, 8)
            rdi_conv_output = rdi_conv_output.view(batch_size, -1)  # 攤平 (batch_size, 64*8*8)
            
            phd_frame = phd_map[:, t, :, :].unsqueeze(1)
            phd_conv_output = self.conv(phd_frame)  # (batch_size, 64, 8, 8)
            phd_conv_output = phd_conv_output.view(batch_size, -1)  # 展平 (batch_size, 64*8*8)
            
            # 將兩個卷積輸出並在一起
            combined_feature = torch.cat((rdi_conv_output, phd_conv_output), dim=1)  # (batch_size, flattened_size*2)
            combined_features.append(combined_feature)
        
        # 把所有時間點的特徵結合成一個序列 (batch_size, timesteps, flattened_size*2)
        combined_features = torch.stack(combined_features, dim=1)

        # 把特徵序列輸入到LSTM中
        lstm_out, _ = self.lstm(combined_features, (h0, c0))
        
        # 取 LSTM 最後的輸出
        lstm_out_last = lstm_out[:, -1, :]
        
        # 全連接層
        out = self.fc(lstm_out_last)
        
        return out
    
## model = initial() 可以用來初始化模型
def initial_model():
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

## result = predict_result(rdi_map, phd_map, model) 用來輸出預測結果
def predict_result(rdi_map, phd_map, model):

    rdi_map = torch.tensor(np.array(rdi_map), dtype=torch.float32).unsqueeze(0).to(device)
    phd_map = torch.tensor(np.array(phd_map), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(rdi_map, phd_map)
        _, predicted_class = torch.max(output, 1)
        print(output)

    predicted_label = label_map_invert[predicted_class.item()] 
    return predicted_label


# 要分類的標籤
label_list = ['background','focus', 'turn_left', 'turn_right', 'turn_up', 'turn_down', 'zoom_in', 'zoom_out']
# 標籤對應字典
label_map = {label: idx for idx, label in enumerate(label_list)}
label_map_invert = {idx: label for idx, label in enumerate(label_list)}

model = initial_model()


def connect():
    connect = ConnectDevice()
    connect.startUp()  # 連接到設備
    reset = ResetDevice()
    reset.startUp()  # 重置硬體暫存器

def startSetting():
    SettingConfigs.setScriptDir("K60168-Test-00256-008-v0.0.8-20230717_60cm")  # 設置腳本目錄
    ksp = SettingProc()  # 設置過程對象
    ksp.startUp(SettingConfigs)  # 啟動設置過程

def startLoop():
    R = FeatureMapReceiver(chirps=32)
    R.trigger(chirps=32)  # 觸發接收器
    time.sleep(0.5)

    print('# ======== Start getting gesture ===========')
    buffer_rdi = []
    buffer_phd = []
    while True :
        res = R.getResults()
        if res is None:
            continue
        if len(buffer_rdi) == 100:    
            rdi_map = buffer_rdi
            phd_map = buffer_phd
            print(f'Detected Gesture: {predict_result(rdi_map, phd_map, model)}')
            buffer_rdi = []
            buffer_phd = []
        else:
            buffer_rdi.append(res[0])
            buffer_phd.append(res[1])


def main():
    kgl.setLib()  # 設置庫
    connect()  # 連接設備
    startSetting()  # 設置設備
    startLoop()  # 開始數據獲取和預測

if __name__ == '__main__':
    main()
