 Renaissance Text Synthesizer (with Degradation)

TL;DR
This repo contains a mid-scale generative model (U-Net + DDIM) for synthesizing Renaissance-style Spanish printed text, complete with historical imperfections: ink bleed, smudging, and fading. The goal: generate convincing degraded text images from 17th-century Spanish content using modern diffusion techniques.
🧠 Model
• Architecture: U-Net (encoder-decoder with skip connections)
• Diffusion Framework: DDIM (fast sampling, fewer steps)
• Training Objective: Learn to reconstruct degraded Renaissance-style text from noise
🛠️ Pipeline Overview 
[Historical Spanish Text]
        ↓
[Font Rendering (Renaissance-style)]
        ↓
[Training Data: Clean ↔ Degraded]
        ↓
[U-Net + DDIM Training]
        ↓
[Sampled Degraded Images]
Directory structure
├── data/
│   ├── clean/        # Rendered clean text images (no degradation)
│   └── degraded/     # Text images with ink bleed, smudging, fading
│
├── outputs/          # Final generated samples and evaluation metrics
│
├── scripts/          # Training, sampling, preprocessing scripts
│
├── utils/            # Helper functions and shared utilities
│
├── README.md         # Project documentation
└── requirements.txt  # Dependency list
📈 Evaluation
Realism isn't easy to quantify—but we tried:
• LPIPS: Perceptual similarity, favors realism over pixel accuracy
• SSIM: Structural fidelity, useful for edge/text clarity
• (Optional) Human Rating
🖼️ Results
Generated outputs convincingly mimic:
• Fuzzy ink in tight curves
• Gradient-like fading toward margins
• Inconsistent smudge directionality
🔍 Why DDIM?
I tried GANs. Required a lot of data to train and very unstable. I wanted control, repeatability, and graceful degradation—not mode collapse. DDIM gives us:
• Smooth interpolation
• Faster sampling
• More predictable noise schedules
📚 Dataset
I used a curated sample of 17th-century Spanish text, formatted via:
• Renaissance.ttf-style fonts
• Custom scripts for rendering 
• Clean/degraded pair generation with seed control
🚧 Limitations & Next Steps
• No multi-column formatting yet
• Augmentations could benefit from learned degradation (e.g. adversarial ink smudge)
• Would like to compare against Stable Diffusion + LoRA fine-tuning
🧾 References
• DDIM: Denoising Diffusion Implicit Models (https://arxiv.org/abs/2010.02502)
• U-Net: Convolutional Networks for Biomedical Image Segmentation (https://arxiv.org/abs/1505.04597)

