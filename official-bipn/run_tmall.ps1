Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force check_point, log\model | Out-Null
& "$PSScriptRoot\..\bipn\.venv\Scripts\python.exe" "$PSScriptRoot\main.py" `
    --lr 0.0001 `
    --reg_weight 0.0001 `
    --log_reg 0.3 `
    --data_name tmall `
    --embedding_size 64 `
    --device cuda:0 `
    --decay 0 `
    --batch_size 1024 `
    --gpu_no 1 `
    --model_name model
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
