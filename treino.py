import os
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# Importe a biblioteca PEFT para LoRA, que é o padrão moderno
from peft import LoraConfig
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def parse_args():
    parser = argparse.ArgumentParser(description="Script de treino LoRA estável.")
    parser.add_argument("--train_data_dir", type=str, required=True, help="Diretório com as imagens e arquivos .txt de legenda.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", type=str, default="./lora_out", help="Diretório para salvar os pesos LoRA treinados.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Taxa de aprendizado. 1e-5 é um bom ponto de partida.")
    parser.add_argument("--max_train_steps", type=int, default=1500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    return parser.parse_args()

class ImageCaptionDataset(Dataset):
    def __init__(self, folder, tokenizer, resolution=512):
        self.folder = folder
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.folder, img_name)
        txt_path = os.path.join(self.folder, base + ".txt")

        img = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(img)

        caption = ""
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        
        # Tokenize o caption aqui
        inputs = self.tokenizer(caption, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt")
        return {"pixel_values": pixel_values, "input_ids": inputs.input_ids.squeeze()}

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if args.mixed_precision == 'fp16' else torch.float32
    print(f"Dispositivo: {device}, Precisão: {args.mixed_precision}")

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype).to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", torch_dtype=weight_dtype).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Configuração LoRA com PEFT (o método moderno)
    lora_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, target_modules=["to_q", "to_v"], lora_dropout=0.1)
    unet.add_adapter(lora_config)
    
    lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
    print(f"Parâmetros treináveis (LoRA): {sum(p.numel() for p in lora_layers)}")

    # <<< MUDANÇA: Otimizador mais estável para fp16
    optimizer = torch.optim.AdamW(lora_layers, lr=args.learning_rate, eps=1e-4)

    dataset = ImageCaptionDataset(args.train_data_dir, tokenizer, resolution=args.resolution)
    # <<< MUDANÇA: num_workers=0 para estabilidade no Windows
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)

    global_step = 0
    pbar = tqdm(total=args.max_train_steps, desc="Passos")

    while global_step < args.max_train_steps:
        for batch in dataloader:
            if global_step >= args.max_train_steps: break
            
            unet.train()
            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            with torch.amp.autocast(device_type="cuda", enabled=(args.mixed_precision == "fp16")):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(lora_layers, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            global_step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item()*args.gradient_accumulation_steps:.5f}"})

            if global_step > 0 and (global_step % args.save_steps == 0 or global_step >= args.max_train_steps):
                print(f"\nSalvando LoRA em {args.output_dir} (passo {global_step})...")
                os.makedirs(args.output_dir, exist_ok=True)
                unet.save_attn_procs(args.output_dir, safe_serialization=True)
                print("Salvo.")
                
    print("Treino finalizado.")

if __name__ == "__main__":
    main()
