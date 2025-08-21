import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,  64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d(64,    3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.main(z)

import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            
            nn.Conv2d(3,  64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 → 128
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 → 256
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 256 → 1
            nn.Conv2d(256,   1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # returns [batch,1]
        return self.main(x).view(-1, 1)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100


gen = Generator(latent_dim).to(device)
gen.load_state_dict(torch.load("generator_now.pth", map_location=device))
gen.eval()

disc = Discriminator().to(device)
disc.load_state_dict(torch.load("discriminator_now.pth", map_location=device))
disc.eval()


CAP_MODEL = "nlpconnect/vit-gpt2-image-captioning"
processor = ViTImageProcessor.from_pretrained(CAP_MODEL)
tokenizer = AutoTokenizer.from_pretrained(CAP_MODEL)
captioner = VisionEncoderDecoderModel.from_pretrained(CAP_MODEL).to(device)
captioner.config.pad_token_id = tokenizer.eos_token_id
captioner.eval()

def make_caption(frame_rgb):
    
    pixel_values = processor(images=frame_rgb, return_tensors="pt").pixel_values.to(device)
    output_ids  = captioner.generate(pixel_values, max_length=32, num_beams=4)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


cap = cv2.VideoCapture(0)
print("🎥 Webcam + real caption + GAN demo. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    disp = cv2.resize(frame, (512, 384))
    rgb  = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

    
    caption = make_caption(rgb)

    
    z = torch.randn(1, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        fake = gen(z)
        score = float(disc(fake))

    fake_img = fake[0].cpu().permute(1,2,0).numpy()
    fake_img = ((fake_img + 1)/2 * 255).astype(np.uint8)
    fake_img = cv2.resize(fake_img, (512,384))

    
    cv2.putText(disp,    f"Caption: {caption}",       (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(fake_img, f"Realism: {score:.2f}",     (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)

    
    out = np.hstack([disp, fake_img])
    cv2.imshow("REAL‑CAPTION | GAN‑OUTPUT", out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
