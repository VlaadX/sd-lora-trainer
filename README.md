![Capa do Projeto](https://raw.githubusercontent.com/cloneofsimo/lora/master/README.md)

Treinar adaptações LoRA (Low-Rank Adaptation) para o UNet de modelos Stable Diffusion, usando pares imagem + legenda (prompt) como dataset. O objetivo é produzir pesos compactos que adaptem o modelo base para um conjunto específico de imagens sem ter que ajustar todo o modelo pré-treinado.

---
## **1.0 Problema e Motivação**

Modelos de difusão pré-treinados (ex.: Stable Diffusion) são grandes e caros para treinar do zero. LoRA permite adaptar esses modelos para domínios específicos (estilos, objetos, conjuntos de dados) de forma eficiente em parâmetros e memória.

Vantagens do LoRA para este projeto:

- Treinar apenas uma pequena fração dos parâmetros (rápido e barato).
- Arquivos de saída pequenos (`.safetensors` / attn_procs) fáceis de versionar.
- Compatível com pipelines Diffusers para inferência direta.

---
## **2.0 Dataset e Pré-processamento**

O repositório inclui uma pasta `Training/` com pares de arquivos `NNN.png|.jpg|.jpeg` e `NNN.txt` (mesmo nome base). O `.txt` contém o prompt/legenda correspondente.

Formato esperado:

```
Training/
  000.png
  000.txt   # legenda em UTF-8
  001.jpg
  001.txt
  ...
```

Pré-processamento realizado pelo `treino.py`:

- Redimensionamento para `--resolution` (padrão 512) com BICUBIC.
- Normalização para o intervalo usado pelo VAE/UNet.
- Tokenização dos prompts com o tokenizer CLIP do modelo base.
- Divisão em batches; `DataLoader` usa `num_workers=0` por compatibilidade com Windows.

Observações: mantenha as legendas curtas e representativas do conteúdo visual; prompts mais informativos ajudam o ajuste fino.

---
## **3.0 Estrutura do Script e Fluxo de Treinamento**

Arquivo principal: `treino.py`

Fluxo resumido:

1. Carrega tokenizer, text encoder, VAE, UNet e scheduler do checkpoint pré-treinado (ex.: `runwayml/stable-diffusion-v1-5`).
2. Congela parâmetros do VAE/TextEncoder/UNet e adiciona adapters LoRA no UNet (módulos `to_q`, `to_v`).
3. Cria o dataset (imagens + prompts), DataLoader e otimizador (AdamW sobre parâmetros LoRA).
4. Para cada batch: codifica latentes com o VAE, aplica ruído, treina o UNet para prever ruído (MSE), acumula gradientes se necessário.
5. Salva os attention processors / pesos LoRA em `--output_dir` a cada `--save_steps`.

Parâmetros importantes (defaults comentados):

- `--train_data_dir` (obrigatório) - ex.: `./Training`
- `--pretrained_model_name_or_path` (default: `runwayml/stable-diffusion-v1-5`)
- `--output_dir` (default: `./lora_out`)
- `--resolution` (default: 512)
- `--train_batch_size` (default: 1)
- `--learning_rate` (default: 1e-5)
- `--max_train_steps` (default: 1500)
- `--save_steps` (default: 500)
- `--lora_rank` (default: 16)
- `--mixed_precision` (`no`|`fp16`|`bf16`, default `fp16`)

---
## **4.0 Comparação de Abordagens e Recomendações**

Em vez de apresentar métricas específicas (dependem do dataset), aqui estão comparações práticas entre estratégias de adaptação:

- Full fine-tuning: ajusta todos os parâmetros; potencialmente o melhor desempenho, porém requer muito GPU/tempo e gera checkpoints enormes.
- LoRA (recomendado): reduz drasticamente o custo computacional e gera arquivos pequenos; ideal para adaptar modelos a conjuntos de imagens específicos.
- DreamBooth / Textual Inversion: são alternativas para casos muito específicos (uma entidade/identidade); DreamBooth treina mais componentes e é custoso.

Recomendação: Para a maioria dos casos de personalização com poucas imagens, use LoRA com `lora_rank` entre 4 e 32 e poucos milhares de passos.

Edge-cases a considerar:

- Dataset muito pequeno (<10 imagens): aumenta overfitting; use augmentations ou prompts variados.
- Alta variabilidade visual: pode exigir mais passos ou rank maior.
- Memória GPU limitada: use `--mixed_precision fp16` e `--gradient_accumulation_steps`.

---
## **5.0 Estrutura do Projeto e Tecnologias**

Tecnologias principais: Python, PyTorch, Hugging Face Diffusers, Transformers, PEFT (LoRA), safetensors, OpenCV.

Estrutura de arquivos (resumida):

```
.
├─ treino.py                # Script de treinamento principal
├─ Training/                # pasta com imagens + .txt
├─ meu_lora/                # exemplo de saída contendo `pytorch_lora_weights.safetensors`
```

## **6.0 Como Executar (PowerShell)**

1) Criar e ativar ambiente virtual e instalar dependências:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r .\lora\requirements.txt
```

2) Rodar o treinamento (exemplo mínimo):

```powershell
python .\treino.py --train_data_dir .\Training --output_dir .\meu_lora_out --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --resolution 512 --train_batch_size 1 --max_train_steps 1500 --save_steps 500 --lora_rank 16 --mixed_precision fp16
```

---

## Compatibilidade com AUTOMATIC1111

Os pesos LoRA gerados por `treino.py` são compatíveis com GUIs populares como o AUTOMATIC1111, desde que você copie o arquivo `.safetensors` (ou a pasta de `attn_procs` convertida) para a pasta correta do webui.

- Se o `--output_dir` gerar um arquivo `.safetensors` (ex.: `pytorch_lora_weights.safetensors`), copie-o para `models\Lora` dentro da instalação do AUTOMATIC1111 e reinicie a interface:

```powershell
Copy-Item .\meu_lora_out\*.safetensors -Destination "C:\caminho\para\stable-diffusion-webui\models\Lora\" -Force
# Depois reinicie o webui e ative o LoRA no painel de geração.
```

- Se o script salvar uma pasta `attn_procs` (formato Diffusers), algumas versões do webui podem exigir conversão para `.safetensors` ou suporte via plugins; verifique a documentação do seu fork.

Observação: caminhos e suporte podem variar entre forks do AUTOMATIC1111; sempre cheque a documentação do seu webui.

## **7.0 Dicas para Windows e Troubleshooting**

- DataLoader / multiprocess hang: `treino.py` usa `num_workers=0` para evitar problemas no Windows.
- Erros de memória CUDA: reduza batch size, aumente `gradient_accumulation_steps` ou use `--mixed_precision fp16`.
- Problemas com drivers CUDA: verifique versão do CUDA e compatibilidade com a versão do PyTorch instalada.
- Cachê do Hugging Face: se houver problemas de I/O, confira permissões em `%USERPROFILE%\\.cache\\huggingface`.

