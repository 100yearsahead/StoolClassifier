import torch
from torchvision import models, transforms
from PIL import Image

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)  # 4 classes
model.load_state_dict(torch.load('model/resnet18_medical.pth'))
model = model.to(device)
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load and preprocess image
image_path = 'path_to_image.jpg'
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0).to(device)

# Predict
outputs = model(image)
_, preds = torch.max(outputs, 1)
classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
print(f'Predicted class: {classes[preds[0]]}')
