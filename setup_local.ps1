# setup_local.ps1
# Post-install specifique Windows + CUDA 12.8
# Usage : conda activate chap10llm && .\setup_local.ps1

Write-Host "=== 1. Torch CUDA 12.4 ===" -ForegroundColor Cyan
pip install torch==2.6.0+cu124 torchvision --index-url https://download.pytorch.org/whl/cu124 

Write-Host "=== 2. Fix OpenMP (KMP_DUPLICATE_LIB_OK) ===" -ForegroundColor Cyan
$site = python -c "import site; print(site.getsitepackages()[0])"
$content = 'import os' + "`n" + 'os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"'
Set-Content -Path "$site\sitecustomize.py" -Value $content -Encoding UTF8
Write-Host "sitecustomize.py cree dans : $site" -ForegroundColor Green

Write-Host "=== 3. Verification ===" -ForegroundColor Cyan
python -c "import torch; print('torch :', torch.__version__); print('CUDA  :', torch.cuda.is_available())"

Write-Host "=== Setup termine ===" -ForegroundColor Green