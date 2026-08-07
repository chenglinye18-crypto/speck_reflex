# Environment setup

Under WSL2, the Windows NVIDIA driver exposes the GPU; do not install a Linux NVIDIA display driver inside WSL. CUDA Toolkit 12.8 provides developer tools, while PyTorch's `cu128` wheel supplies its matched runtime libraries—these are distinct.

Install Torch 2.10.0, torchvision 0.25.0, and torchaudio 2.10.0 from `https://download.pytorch.org/whl/cu128` before `requirements/core.txt`. The verified software baseline uses Python 3.10, Sinabs 3.1.3, Samna 0.48.6, Tonic 1.6.0, and NumPy 1.26.4. `requirements/environment-cu128-reference.txt` is a machine freeze, not an installer.

USB forwarding (`usbipd`) and udev configuration are intentionally deferred until a physical board exists and hardware access is explicitly approved.
