import torch as th
import torch.nn as nn
import torchvision.models as models
from logger import logger
from gymnasium import spaces
from ultralytics import YOLO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MultiTaskCNN(BaseFeaturesExtractor):
    '''
    Multi-task CNN model with ResNet-50 backbone for feature extraction,
    and separate heads for detection and segmentation tasks.
    '''
    def __init__(self, observation_space, features_dim=256):
        super(MultiTaskCNN, self).__init__(observation_space, features_dim)
        # Using ResNet-50 as the backbone
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity() # Remove the last fully connected layer
        # detection head
        self.detector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim), 
        )
    
        # segmentation head
        self.segmenter = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )
        # self.segmentation_fc = nn.Linear(8 * 8, 256)

        self.combined_fc = nn.Sequential(
            nn.Linear(features_dim + 64 + 2, features_dim), # [detection_features = 256, segmentation_features = 64, steering_speed = 2]
            nn.ReLU(),
        )
        
        if th.cuda.is_available():
            self.device = th.device("cuda")
            logger.info("CUDA is available. Using CUDA instead.")
        else:
            self.device = th.device("cpu")
            logger.info("CUDA is not available. Using CPU instead.")
        self.to(self.device)

    def forward(self, observations):


        # 提取圖像輸入和非圖像輸入 (steering_speed)
        image = observations['image'].to(self.device)
        image = image.repeat(1, 3, 1, 1)
        steering_speed = observations['steering_speed'].to(self.device)

        # Backbone 提取圖像特徵
        features = self.backbone(image)
        detection_features = self.detector(features)
        
        # Add two dimensions to features to match the segmentation head
        features = features.unsqueeze(-1).unsqueeze(-1)
        segmentation_output = self.segmenter(features)

        segmentation_features = segmentation_output.flatten(start_dim=1).to(self.device) # [1,1,8,8]

        # segmentation_features = self.segmentation_fc(segmentation_features)
        
        combined_features = th.cat([detection_features, segmentation_features, steering_speed], dim=1) # 256 8 2 

        return self.combined_fc(combined_features)

class MultiTaskRNN(BaseFeaturesExtractor):
    '''
    redundant to have additional detector and segmenter
    Multi-task RNN model with YOLOv8 backbone for feature extraction,
    and an LSTM for sequential modeling.
    '''
    def __init__(self, observation_space, features_dim=256, hidden_dim=512, num_layers=1):
        super(MultiTaskRNN, self).__init__(observation_space, features_dim)
        
        # Initialize YOLOv8 model and extract the backbone
        yolo_model = YOLO('yolov8n.pt')  # Using the nano version for example
        self.backbone = yolo_model.model.backbone
        
        # Optionally freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False  # Set to True if you want to fine-tune
        
        # Determine the output dimension of the backbone
        with th.no_grad():
            dummy_input = th.zeros(1, 3, 640, 640)  # Adjust input size as needed
            backbone_outputs = self.backbone(dummy_input)
            # Concatenate outputs from different stages
            backbone_output = th.cat([f.flatten(1) for f in backbone_outputs], dim=1)
            backbone_output_dim = backbone_output.shape[1]
        
        # LSTM for sequential modeling
        self.lstm = nn.LSTM(input_size=backbone_output_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        
        # Detection head
        self.detector = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
        )
        
        # Segmentation head
        self.segmenter = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        
        # Final combined fully connected layer
        self.combined_fc = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )
        
        # Device setup
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, observations):
        # Extract image sequence and steering_speed (if applicable)
        image_seq = observations['image'].to(self.device)  # Shape: [batch_size, seq_len, C, H, W]
        
        batch_size, seq_len, C, H, W = image_seq.shape
        
        # Reshape to process through the backbone
        images = image_seq.view(batch_size * seq_len, C, H, W)
        
        # Extract features using the YOLOv8 backbone
        with th.no_grad():
            backbone_features = self.backbone(images)
        
        # Concatenate features from different stages
        features = th.cat([f.flatten(1) for f in backbone_features], dim=1)  # Shape: [batch_size * seq_len, feature_dim]
        
        # Reshape features back to sequence format
        features_seq = features.view(batch_size, seq_len, -1)
        
        # Pass the sequence through the LSTM
        lstm_output, _ = self.lstm(features_seq)
        
        # Use the last output from the LSTM
        lstm_last_output = lstm_output[:, -1, :]  # Shape: [batch_size, hidden_dim]
        
        # Detection features
        detection_features = self.detector(lstm_last_output)
        
        # Segmentation output
        segmentation_input = lstm_last_output.view(batch_size, -1, 1, 1)  # Reshape for ConvTranspose2d
        segmentation_output = self.segmenter(segmentation_input)
        
        # Final combined features (you can modify this as needed)
        combined_features = self.combined_fc(detection_features)
        
        return combined_features  # Return the combined features or a tuple if you need multiple outputs

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from ultralytics import YOLO

class MultiTaskYOLORNN(BaseFeaturesExtractor):
    '''
    Multi-task RNN model with YOLOv8 backbone for feature extraction,
    and an LSTM for sequential modeling.
    '''
    def __init__(self, observation_space, features_dim=256, hidden_dim=512, num_layers=1):
        super(MultiTaskRNN, self).__init__(observation_space, features_dim)
        
        # Initialize YOLOv8 model and extract the backbone
        yolo_model = YOLO('yolov8n.pt')  # Using the nano version for example
        self.backbone = yolo_model.model.backbone
        
        # Optionally freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False  # Set to True if you want to fine-tune
        
        # Determine the output dimension of the backbone
        with th.no_grad():
            dummy_input = th.zeros(1, 3, 640, 640)  # Adjust input size as needed
            backbone_outputs = self.backbone(dummy_input)
            # Concatenate outputs from different stages
            backbone_output = th.cat([f.flatten(1) for f in backbone_outputs], dim=1)
            backbone_output_dim = backbone_output.shape[1]
        
        # LSTM for sequential modeling
        self.lstm = nn.LSTM(input_size=backbone_output_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        
        # Detection head
        self.detector = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
        )
        
        # Segmentation head
        self.segmenter = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        
        # Final combined fully connected layer
        self.combined_fc = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )
        
        # Device setup
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, observations):
        # Extract image sequence and steering_speed (if applicable)
        image_seq = observations['image'].to(self.device)  # Shape: [batch_size, seq_len, C, H, W]
        
        batch_size, seq_len, C, H, W = image_seq.shape
        
        # Reshape to process through the backbone
        images = image_seq.view(batch_size * seq_len, C, H, W)
        
        # Extract features using the YOLOv8 backbone
        with th.no_grad():
            backbone_features = self.backbone(images)
        
        # Concatenate features from different stages
        features = th.cat([f.flatten(1) for f in backbone_features], dim=1)  # Shape: [batch_size * seq_len, feature_dim]
        
        # Reshape features back to sequence format
        features_seq = features.view(batch_size, seq_len, -1)
        
        # Pass the sequence through the LSTM
        lstm_output, _ = self.lstm(features_seq)
        
        # Use the last output from the LSTM
        lstm_last_output = lstm_output[:, -1, :]  # Shape: [batch_size, hidden_dim]
        
        # Detection features
        detection_features = self.detector(lstm_last_output)
        
        # Segmentation output
        segmentation_input = lstm_last_output.view(batch_size, -1, 1, 1)  # Reshape for ConvTranspose2d
        segmentation_output = self.segmenter(segmentation_input)
        
        # Final combined features (you can modify this as needed)
        combined_features = self.combined_fc(detection_features)
        
        return combined_features  # Return the combined features or a tuple if you need multiple outputs


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for handling image input and extracting features.

    Args:
        observation_space (spaces.Dict): The observation space which includes the image input.
        features_dim (int): The dimension of the output feature vector after CNN layers.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # Extract the 'image' shape from observation space, assuming image is (64, 64, 3)
        super(CustomCNN, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space['image'].shape[0]  # Get the number of input channels (stacked frames)

        # Define CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=9, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Get the output dimension of the CNN layers
        with th.no_grad():
            sample_input = th.zeros(1, *observation_space['image'].shape)
            cnn_output_dim = self.cnn(sample_input).shape[1]

        # Define a fully connected layer to combine CNN output with other inputs (steering/speed)
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim + 2, features_dim),  # Add steering and speed (2,)
            nn.ReLU(),
        )

    def forward(self, observations):
        """
        Forward pass for feature extraction.

        Args:
            observations (dict): A dictionary containing 'image' and 'steering_speed' inputs.

        Returns:
            Tensor: A tensor representing extracted features from image and steering/speed.
        """
        image = observations['image']  # Extract image input
        image_features = self.cnn(image)  # Extract features using CNN

        # Process non-image input (steering and speed)
        steering_speed = observations['steering_speed']

        # Concatenate image features and steering/speed, and pass through the linear layer
        return self.linear(th.cat([image_features, steering_speed], dim=1))
    


class MultiTaskYOLO(BaseFeaturesExtractor):
    '''
    尚未測試
    調用YOOLO模型提取特徵
    '''
    def __init__(self, observation_space, features_dim=256):
        super(MultiTaskYOLO, self).__init__(observation_space, features_dim)
        
        # 使用 YOLO 作为骨干网络
        self.yolo_model = YOLO('yolov5s.pt')  # 使用 YOLOv5s 预训练模型
        
        # 冻结 YOLO 的大部分层，只保留最后几层进行训练（可选）
        for param in self.yolo_model.model.parameters():
            param.requires_grad = False
        
        # 解冻最后几层
        for param in self.yolo_model.model[-1].parameters():
            param.requires_grad = True
        
        # 分割头
        self.segmenter = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )
        
        # 融合层
        self.combined_fc = nn.Sequential(
            nn.Linear(features_dim + 64 + 2, features_dim),  # detection_features, segmentation_features, steering_speed
            nn.ReLU(),
        )

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, observations):
        image = observations['image'].to(self.device)
        image = image.repeat(1, 3, 1, 1)
        steering_speed = observations['steering_speed'].to(self.device)

        # 使用 YOLO 提取特征
        yolo_output = self.yolo_model(image, verbose=False)
        detection_features = yolo_output.pred[0]  # 获取 YOLO 的预测结果
        
        # 提取 YOLO 的特征图
        yolo_features = self.yolo_model.model[-2](image)  # 使用 YOLO 的特征层
        
        # 分割输出
        segmentation_output = self.segmenter(yolo_features)
        segmentation_features = segmentation_output.flatten(start_dim=1)

        # 融合特征
        combined_features = th.cat([detection_features, segmentation_features, steering_speed], dim=1)
        
        return self.combined_fc(combined_features)
