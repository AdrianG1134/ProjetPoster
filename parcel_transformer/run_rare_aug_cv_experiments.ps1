$ErrorActionPreference = "Stop"

$csv = "data/s2_herault_2024_full_year_5day_cloudmask_fast/indices_parcelles_2024-01-01_2024-12-31_win5d_with_labels_and_group_min200.csv"
$outRoot = "outputs_transformer/rare_aug_cv_experiments"

$baseArgs = @(
  "--csv-path", $csv,
  "--split-method", "tile",
  "--index-filter", "NDVI,NDMI,NDWI,EVI",
  "--loss-type", "focal",
  "--focal-gamma", "1.5",
  "--class-weighting",
  "--class-weight-power", "0.5",
  "--standardize-features",
  "--no-use-group-task",
  "--no-weighted-sampler"
)

Write-Host "`n=== RUN aug_only ==="
python parcel_transformer/train.py @baseArgs `
  --temporal-augmentation --time-mask-ratio 0.05 --jitter-std 0.01 `
  --output-dir "$outRoot/aug_only"

Write-Host "`n=== RUN phase2_rare_only ==="
python parcel_transformer/train.py @baseArgs `
  --phase2-rare-finetune --phase2-epochs 12 --phase2-lr 1e-4 `
  --phase2-sampler-power 1.0 --phase2-rare-quantile 0.25 --phase2-rare-boost 2.0 `
  --output-dir "$outRoot/phase2_rare_only"

Write-Host "`n=== RUN aug_plus_phase2 ==="
python parcel_transformer/train.py @baseArgs `
  --temporal-augmentation --time-mask-ratio 0.05 --jitter-std 0.01 `
  --phase2-rare-finetune --phase2-epochs 12 --phase2-lr 1e-4 `
  --phase2-sampler-power 1.0 --phase2-rare-quantile 0.25 --phase2-rare-boost 2.0 `
  --output-dir "$outRoot/aug_plus_phase2"

Write-Host "`n=== RUN spatial_cv (3 folds) ==="
python parcel_transformer/spatial_cv_groupkfold.py `
  --csv-path $csv `
  --index-filter "NDVI,NDMI,NDWI,EVI" `
  --n-splits 3 `
  --val-size 0.1 `
  --seed 42 `
  --output-root "$outRoot/spatial_cv" `
  --train-extra-args "--loss-type focal --focal-gamma 1.5 --class-weighting --class-weight-power 0.5 --standardize-features --no-use-group-task --no-weighted-sampler"

Write-Host "`n=== SUMMARY (single runs) ==="
$rows = Get-ChildItem "$outRoot" -Recurse -Filter test_metrics.json | ForEach-Object {
  $m = Get-Content $_.FullName | ConvertFrom-Json
  [pscustomobject]@{
    run              = $_.DirectoryName
    test_macro_f1    = [double]$m.f1_macro
    test_accuracy    = [double]$m.accuracy
    test_weighted_f1 = [double]$m.f1_weighted
  }
}
$rows | Sort-Object test_macro_f1 -Descending | Format-Table -AutoSize
$rows | Sort-Object test_macro_f1 -Descending | Export-Csv (Join-Path $outRoot "summary.csv") -NoTypeInformation
Write-Host "[OK] Summary saved to $outRoot/summary.csv"
