import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from util.logger import logger
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
        # image = image.repeat(1, 3, 1, 1)
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

class MultiTaskYOLORNN(BaseFeaturesExtractor):
    '''
    Multi-task RNN model with YOLOv8 backbone for feature extraction,
    and an LSTM for sequential modeling.
    '''
    def __init__(self, observation_space, features_dim=256, hidden_dim=512, num_layers=1):
        super(MultiTaskYOLORNN, self).__init__(observation_space, features_dim)
        
        # Initialize YOLOv8 model and extract the backbone
        yolo_model = YOLO('yolov8n.pt')  # Using the nano version for example
        self.backbone = yolo_model.model
        # self.backbone = nn.Sequential(*list(yolo_model.model[:10]))
        
        # Optionally freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False  # Set to True if you want to fine-tune
        
        # # Unfreeze the last few layers for training
        # for param in self.yolo_model.model[-1].parameters(): #TypeError: 'DetectionModel' object is not subscriptable
        #     param.requires_grad = True

        # Determine the output dimension of the backbone
        with th.no_grad():
            dummy_input = th.zeros(1, 3, 128, 128)
            backbone_outputs = self.backbone(dummy_input)
            # Check if backbone outputs are a list or tuple
            if isinstance(backbone_outputs, list) or isinstance(backbone_outputs, tuple):
                # If the output is a list, concatenate the tensors
                backbone_output = th.cat([f.view(f.size(0), -1) for f in backbone_outputs if isinstance(f, th.Tensor)], dim=1)
            elif isinstance(backbone_outputs, th.Tensor):
                # If the output is a single tensor, flatten it
                backbone_output = backbone_outputs.view(backbone_outputs.size(0), -1)
            else:
                # Capture unexpected output types
                raise ValueError(f"Unexpected output type from backbone: {type(backbone_outputs)}")
            backbone_output_dim = backbone_output.shape[1]
            print("Backbone output shape:", backbone_output.shape)
        
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
            nn.Linear(features_dim + 2, features_dim),
            nn.ReLU(),
        )
        
        # Device setup
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, observations):
        # Extract image sequence and steering_speed (if applicable)
        image_seq = observations['image'].to(self.device)  # Expected shape: [batch_size, seq_len, C, H, W]
        batch_size, seq_len, C, H, W = image_seq.shape
        
        # Reshape to process through the backbone
        images = image_seq.view(batch_size * seq_len, C, H, W)
        
        # Extract features using the YOLOv8 backbone
        with th.no_grad():
            backbone_outputs = self.backbone(images)
        
        # Concatenate features from different stages
        if isinstance(backbone_outputs, (list, tuple)):
            features = th.cat([f.flatten(1) for f in backbone_outputs], dim=1)
        else:
            features = backbone_outputs.flatten(1)
        
        # Reshape features back to sequence format
        features_seq = features.view(batch_size, seq_len, -1)
        
        # Pass the sequence through the LSTM
        lstm_output, _ = self.lstm(features_seq)
        
        # Use the last output from the LSTM
        lstm_last_output = lstm_output[:, -1, :]  # Shape: [batch_size, hidden_dim]
        
        # Detection features
        detection_features = self.detector(lstm_last_output)
        
        # Segmentation output
        # segmentation_input = lstm_last_output.view(batch_size, -1, 1, 1)  # Reshape for ConvTranspose2d
        # segmentation_output = self.segmenter(segmentation_input)
        
        # If you have steering_speed input, process it here
        steering_speed = observations['steering_speed'].to(self.device)  # Shape: [batch_size, 2]
        combined_features = th.cat([detection_features, steering_speed], dim=1)
        
        # Final combined features
        combined_features = self.combined_fc(combined_features)
        
        return combined_features  # You can also return segmentation_output if needed

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

class CustomYOLOv5n(BaseFeaturesExtractor):
    """
    Custom feature extractor using YOLOv5n's pre-trained backbone for image input.

    Args:
        observation_space (spaces.Dict): The observation space which includes the image input.
        features_dim (int): The dimension of the output feature vector after YOLOv5n's backbone layers.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # Initialize with the complete observation space
        super(CustomYOLOv5n, self).__init__(observation_space, features_dim)

        # Get the image shape from the observation space
        if 'image' not in observation_space.spaces:
            raise ValueError("No 'image' key in observation space")
        
        # Get image dimensions
        image_space = observation_space.spaces['image']
        if not isinstance(image_space, spaces.Box):
            raise ValueError("Image space must be of type Box")
            
        image_shape = image_space.shape
        print(f"Input image shape: {image_shape}")  # Debugging info

        # Determine input channels (should be the last dimension for gym/gymnasium)
        n_input_channels = image_shape[0]
        self.input_height = image_shape[1]
        self.input_width = image_shape[2]

        # Load YOLOv5n model
        yolo_model = th.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        

        # Extract the backbone (remove detection head)
        self.backbone = nn.Sequential(*list(yolo_model.model.children())[:-1])

        # Calculate the output dimension of the backbone
        with th.no_grad():
            # Create a sample input with the correct shape
            sample_input = th.zeros(1, n_input_channels, self.input_height, self.input_width)
            backbone_output = self.backbone(sample_input)
            if isinstance(backbone_output, list):
                backbone_output = backbone_output[-1]
            cnn_output_dim = backbone_output.view(backbone_output.size(0), -1).shape[1]

        # Define the feature processing layers
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU(),
        )

        # Final combined layer for image features and steering/speed
        self.combined_fc = nn.Sequential(
            nn.Linear(features_dim + 2, features_dim),  # +2 for steering and speed
            nn.ReLU(),
        )

        # Set up device
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, observations):
        """
        Forward pass for feature extraction.

        Args:
            observations (dict): A dictionary containing 'image' and 'steering_speed' inputs.

        Returns:
            Tensor: Extracted features for downstream tasks.
        """
        # Process image input
        image = observations['image'].to(self.device)
        
        # Ensure correct shape (N, C, H, W)
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension if missing
        image = image.permute(0, 3, 1, 2)  # Move channels to second dimension
        
        # Extract features using backbone
        yolo_features = self.backbone(image)
        yolo_features = yolo_features.contiguous()
        yolo_flattened = yolo_features.view(yolo_features.size(0), -1)
        # Process steering and speed
        steering_speed = observations['steering_speed'].to(self.device)

        # Combine features
        yolo_processed = self.linear(yolo_flattened)
        combined = th.cat([yolo_processed, steering_speed], dim=1)
        
        return self.combined_fc(combined)