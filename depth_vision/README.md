# Depth Vision - Multi-Model Framework

Monocular depth estimation using multiple state-of-the-art models:
- **MiDaS** - Intel's robust depth estimation model
- **Depth Anything V2** - Latest high-performance depth estimator
- **ZoeDepth** - High-quality metric depth estimation
- **Marigold** - Diffusion-based depth estimation with fine details

## Setup

## Available Models

### MiDaS (Intel)
- `DPT_Large` - Highest accuracy, ~1.3GB
- `DPT_Hybrid` - Balanced accuracy/speed, ~800MB (recommended)
- `MiDaS_small` - Fastest, ~80MB

### Depth Anything V2
- `small` - Fast inference, ~100MB
- `base` - Balanced, ~400MB  
- `large` - Highest accuracy, ~1.3GB

### ZoeDepth
- `NK` - General purpose (indoor + outdoor), ~350MB (recommended)
- `N` - Indoor scenes (NYU dataset), ~350MB
- `K` - Outdoor/driving scenes (KITTI dataset), ~350MB

### Marigold
- `lcm` - Fast diffusion model with Latent Consistency, ~2GB (recommended)
- `base` - Standard diffusion model (slower but highly accurate), ~2GB

| Configuration           | Quality  | Speed  | Use Case                        |
|------------------------|----------|--------|---------------------------------|
| ensemble=1, steps=10    | Basic    | ~17s   | Fast preview                    |
| ensemble=1, steps=50    | Good     | ~70s   | Single prediction, full quality |
| ensemble=5, steps=50    | Best     | ~6min  | Production, highest accuracy    |
| ensemble=10, steps=50   | Maximum  | ~12min | Research, best possible quality |