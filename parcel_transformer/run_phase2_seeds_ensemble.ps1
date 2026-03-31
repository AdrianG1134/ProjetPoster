$ErrorActionPreference = "Stop"

$csv = "data/s2_herault_2024_full_year_5day_cloudmask_fast/indices_parcelles_2024-01-01_2024-12-31_win5d_with_labels_and_group_min200.csv"
$outRoot = "outputs_transformer/phase2_seeds_ensemble"
$seeds = @(42, 123, 777)

$commonArgs = @(
  "--csv-path", $csv,
  "--split-method", "tile",
  "--index-filter", "NDVI,NDMI,NDWI,EVI",
  "--loss-type", "focal",
  "--focal-gamma", "1.5",
  "--class-weighting",
  "--class-weight-power", "0.5",
  "--standardize-features",
  "--no-use-group-task",
  "--no-weighted-sampler",
  "--phase2-rare-finetune",
  "--phase2-epochs", "12",
  "--phase2-lr", "1e-4",
  "--phase2-sampler-power", "1.0",
  "--phase2-rare-quantile", "0.25",
  "--phase2-rare-boost", "2.0"
)

foreach ($seed in $seeds) {
  Write-Host "`n=== RUN seed=$seed (phase2_rare_only) ==="
  python parcel_transformer/train.py @commonArgs --seed $seed --output-dir "$outRoot/s$seed"
}

Write-Host "`n=== SUMMARY (single models) ==="
$rows = Get-ChildItem "$outRoot/s*" -Recurse -Filter test_metrics.json | ForEach-Object {
  $m = Get-Content $_.FullName | ConvertFrom-Json
  [pscustomobject]@{
    run              = $_.DirectoryName
    test_macro_f1    = [double]$m.f1_macro
    test_accuracy    = [double]$m.accuracy
    test_weighted_f1 = [double]$m.f1_weighted
  }
}

$rows | Sort-Object test_macro_f1 -Descending | Format-Table -AutoSize
$rows | Sort-Object test_macro_f1 -Descending | Export-Csv (Join-Path $outRoot "single_models_summary.csv") -NoTypeInformation
Write-Host "[OK] Single-model summary saved to $outRoot/single_models_summary.csv"

$ckptGlob = "$outRoot/s*/temporal_transformer_*/best_model.pt"
Write-Host "`n=== ENSEMBLE EVAL ==="
python parcel_transformer/evaluate_ensemble.py `
  --checkpoint-glob $ckptGlob `
  --csv-path $csv `
  --split-method tile `
  --index-filter NDVI,NDMI,NDWI,EVI `
  --output-dir "$outRoot/ensemble_eval"

$ensembleMetricsPath = Join-Path $outRoot "ensemble_eval/test_ensemble_metrics.json"
if (Test-Path $ensembleMetricsPath) {
  $e = Get-Content $ensembleMetricsPath | ConvertFrom-Json
  Write-Host ("[ENSEMBLE] acc={0:N4} macro_f1={1:N4} weighted_f1={2:N4}" -f $e.accuracy, $e.f1_macro, $e.f1_weighted)
} else {
  Write-Host "[WARN] Ensemble metrics file not found: $ensembleMetricsPath"
}
