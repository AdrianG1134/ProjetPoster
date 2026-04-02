param(
  [string]$PreparedNpz = "data/parcel_dataset_ext.npz",
  [string]$EncoderPath = "",
  [string]$SslRoot = "outputs_transformer/ssl_pretrain_ext",
  [string]$OutRoot = "outputs_transformer/ssl_finetune_ext_seeds",
  [int[]]$Seeds = @(42, 123, 777, 2024),
  [switch]$SkipTrain,
  [switch]$SkipEnsemble
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PreparedNpz)) {
  throw "Prepared NPZ not found: $PreparedNpz"
}

if ([string]::IsNullOrWhiteSpace($EncoderPath)) {
  $candidate = Get-ChildItem (Join-Path $SslRoot "ssl_pretrain_*\best_ssl_encoder.pt") -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if (-not $candidate) {
    throw "No SSL encoder found under $SslRoot. Provide -EncoderPath explicitly."
  }
  $EncoderPath = $candidate.FullName
}

if (-not (Test-Path $EncoderPath)) {
  throw "Encoder checkpoint not found: $EncoderPath"
}

Write-Host "[INFO] NPZ: $PreparedNpz"
Write-Host "[INFO] Encoder: $EncoderPath"
Write-Host "[INFO] OutRoot: $OutRoot"
Write-Host "[INFO] Seeds: $($Seeds -join ', ')"

if (-not $SkipTrain) {
  foreach ($seed in $Seeds) {
    $seedOut = Join-Path $OutRoot ("s{0}" -f $seed)
    Write-Host ""
    Write-Host "=== RUN seed=$seed ==="
    python parcel_transformer/train.py `
      --prepared-npz $PreparedNpz `
      --split-method tile `
      --reliability-aware `
      --pretrained-encoder-checkpoint $EncoderPath `
      --seed $seed `
      --loss-type focal `
      --focal-gamma 1.5 `
      --class-weighting `
      --class-weight-power 0.5 `
      --standardize-features `
      --no-use-group-task `
      --phase2-rare-finetune `
      --phase2-epochs 12 `
      --phase2-lr 1e-4 `
      --phase2-sampler-power 1.0 `
      --phase2-rare-quantile 0.25 `
      --phase2-rare-boost 2.0 `
      --batch-size 48 `
      --output-dir $seedOut
  }
}

if (-not $SkipEnsemble) {
  $ckptGlob = (Join-Path $OutRoot "s*/temporal_transformer_*/best_model.pt")

  Write-Host ""
  Write-Host "=== RUN ensemble_uniform ==="
  python parcel_transformer/evaluate_ensemble.py `
    --checkpoint-glob $ckptGlob `
    --prepared-npz $PreparedNpz `
    --ensemble-weighting uniform `
    --output-dir (Join-Path $OutRoot "ensemble_uniform")

  Write-Host ""
  Write-Host "=== RUN ensemble_weighted ==="
  python parcel_transformer/evaluate_ensemble.py `
    --checkpoint-glob $ckptGlob `
    --prepared-npz $PreparedNpz `
    --ensemble-weighting val_macro_f1 `
    --weight-power 1.0 `
    --output-dir (Join-Path $OutRoot "ensemble_weighted")
}

Write-Host ""
Write-Host "=== SUMMARY (single models) ==="
$rows = @()
foreach ($seed in $Seeds) {
  $metric = Get-ChildItem (Join-Path $OutRoot ("s{0}" -f $seed)) -Recurse -Filter test_metrics.json -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if ($metric) {
    $m = Get-Content $metric.FullName | ConvertFrom-Json
    $rows += [PSCustomObject]@{
      seed = $seed
      run = $metric.DirectoryName
      test_macro_f1 = [double]$m.f1_macro
      test_accuracy = [double]$m.accuracy
      test_weighted_f1 = [double]$m.f1_weighted
    }
  }
}

if ($rows.Count -gt 0) {
  $rows | Sort-Object test_macro_f1 -Descending | Format-Table -AutoSize
  $rows | Sort-Object test_macro_f1 -Descending | Export-Csv (Join-Path $OutRoot "single_models_summary.csv") -NoTypeInformation
  Write-Host "[OK] Single-model summary saved to $(Join-Path $OutRoot 'single_models_summary.csv')"
}

$uniformPath = Join-Path $OutRoot "ensemble_uniform/test_ensemble_metrics.json"
if (Test-Path $uniformPath) {
  $u = Get-Content $uniformPath | ConvertFrom-Json
  Write-Host ("[ENSEMBLE uniform]  acc={0:N4} macro_f1={1:N4} weighted_f1={2:N4}" -f $u.accuracy, $u.f1_macro, $u.f1_weighted)
}

$weightedPath = Join-Path $OutRoot "ensemble_weighted/test_ensemble_metrics.json"
if (Test-Path $weightedPath) {
  $w = Get-Content $weightedPath | ConvertFrom-Json
  Write-Host ("[ENSEMBLE weighted] acc={0:N4} macro_f1={1:N4} weighted_f1={2:N4}" -f $w.accuracy, $w.f1_macro, $w.f1_weighted)
}
