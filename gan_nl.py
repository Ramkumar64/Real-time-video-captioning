import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer
)
from datasets import load_from_disk
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

# ------------------- Configuration -------------------
CONFIG = {
    "gan": {
        # Reduced image size and increased batch size for faster training
        "latent_dim": 100,
        "img_size": 128,
        "channels": 3,
        "batch_size": 64,
        "lr": 0.0002,
        # Fewer epochs to fit within 30 mins
        "epochs": 20
    },
    "nlp": {
        "model_name": "nlpconnect/vit-gpt2-image-captioning",
        "max_length": 128,
        "output_dir": "output/captions.txt"
    },
    "paths": {
        "dataset": "flickr8k_dataset",
        "images": "data/Images",
        "utils": "utils"
    }
}

# ------------------- Seed Setup -------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# ------------------- Weight Initialization -------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ------------------- GAN Models -------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(CONFIG['gan']['latent_dim'], 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, CONFIG['gan']['channels'], 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(CONFIG['gan']['channels'], 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        out = self.main(x)
        out = out.view(out.size(0), -1).mean(1, keepdim=True)
        return out

# ------------------- Dataset -------------------
class VideoCaptionDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform or transforms.Compose([
            transforms.Resize((CONFIG['gan']['img_size'], CONFIG['gan']['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*CONFIG['gan']['channels'], [0.5]*CONFIG['gan']['channels'])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img_path = os.path.join(CONFIG['paths']['images'], os.path.basename(item['image_path']))
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        caption = item.get('caption', '')
        return image, caption

# ------------------- Captioning & Training -------------------
class VideoCaptioner:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load vision-language model
        self.processor = ViTImageProcessor.from_pretrained(CONFIG['nlp']['model_name'])
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['nlp']['model_name'])
        self.model = VisionEncoderDecoderModel.from_pretrained(CONFIG['nlp']['model_name'])
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.to(self.device)

        # Initialize GAN components
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        # Preprocess utility fallback
        try:
            from utils.captioning_utils import preprocess_video_frame
            self.preprocess_frame = preprocess_video_frame
        except ImportError:
            self.preprocess_frame = transforms.Resize((CONFIG['gan']['img_size'], CONFIG['gan']['img_size']))

    def train_gan(self, dataset):
        dataloader = DataLoader(dataset,
                                batch_size=CONFIG['gan']['batch_size'],
                                shuffle=True,
                                drop_last=True)
        g_optim = Adam(self.generator.parameters(), lr=CONFIG['gan']['lr'], betas=(0.5, 0.999))
        d_optim = Adam(self.discriminator.parameters(), lr=CONFIG['gan']['lr'], betas=(0.5, 0.999))
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(CONFIG['gan']['epochs']):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['gan']['epochs']}")
            for real_imgs, _ in pbar:
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)

                # Train Discriminator
                self.discriminator.zero_grad()
                with torch.cuda.amp.autocast():
                    real_pred = self.discriminator(real_imgs)
                    errD_real = criterion(real_pred, real_labels)

                    noise = torch.randn(batch_size,
                                         CONFIG['gan']['latent_dim'],
                                         1, 1,
                                         device=self.device)
                    fake_imgs = self.generator(noise)
                    fake_pred = self.discriminator(fake_imgs.detach())
                    errD_fake = criterion(fake_pred, fake_labels)

                    errD = errD_real + errD_fake
                scaler.scale(errD).backward()
                scaler.step(d_optim)
                scaler.update()

                # Train Generator
                self.generator.zero_grad()
                with torch.cuda.amp.autocast():
                    fake_pred = self.discriminator(fake_imgs)
                    errG = criterion(fake_pred, real_labels)
                scaler.scale(errG).backward()
                scaler.step(g_optim)
                scaler.update()

                pbar.set_postfix(loss_D=errD.item(), loss_G=errG.item())

    def generate_caption(self, frame):
        if hasattr(self.preprocess_frame, '__call__'):
            frame = self.preprocess_frame(frame)
        pixel_values = self.processor(images=frame,
                                      return_tensors="pt").pixel_values.to(self.device)

        output_ids = self.model.generate(
            pixel_values,
            max_length=CONFIG['nlp']['max_length'],
            num_beams=4,
            early_stopping=True
        )
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        os.makedirs(os.path.dirname(CONFIG['nlp']['output_dir']), exist_ok=True)
        with open(CONFIG['nlp']['output_dir'], "a") as f:
            f.write(f"{caption}\n")

        return caption

    def process_video(self, video_path):
        from utils.video_utils import VideoProcessor
        vp = VideoProcessor(video_path)
        for frame in vp.stream():
            yield frame, self.generate_caption(frame)

# ------------------- Entry Point -------------------
def main():
    captioner = VideoCaptioner()
    dataset = load_from_disk(CONFIG['paths']['dataset'])
    if 'train' not in dataset:
        raise KeyError("Dataset does not contain 'train' split")

    train_ds = VideoCaptionDataset(dataset['train'])

    if input("Train GAN? (y/n): ").strip().lower() == 'y':
        captioner.train_gan(train_ds)

    video_path = input("Enter video path: ")
    for frame, caption in captioner.process_video(video_path):
        print(f"Caption: {caption}")

if __name__ == "__main__":
    main()
