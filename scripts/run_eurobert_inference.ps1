# Prevent script from stopping on non-terminating errors
$ErrorActionPreference = "Continue"

# Common arguments
$model_path = "T:\laupodteam\AIOS\Bram\language_modeling\Models\language_models\MultiClinAI\EuroBERT\DISEASE_3ldense_20epochs\fold_0"
$base_corpus_path = "T:\laupodteam\AIOS\Bram\notebooks\code_dev\CardioNER.nl\assets\MultiClinNER_combined"
$filter_file = "T:\laupodteam\AIOS\Bram\notebooks\code_dev\CardioNER.nl\assets\MultiClinNER_combined\batch_1_ids.txt"

# Languages
$languages = @("sv", "ro", "nl", "it", "es", "en", "cz")

# Log file
$logFile = "inference_log.txt"

foreach ($lang in $languages) {
    try {
        $corpus_path = Join-Path $base_corpus_path "MultiClinNER-$lang\test"
        $output_prefix = "DISEASE_EuroBERT610_multilabel_$lang"

        $msg = "[$(Get-Date)] Starting language: $lang"
        Write-Host $msg
        Add-Content $logFile $msg

        poetry run python -m cardioner.main `
            --inference_only `
            --model_path=$model_path `
            --inference_pipe=dt4h `
            --corpus_inference=$corpus_path `
            --inference_batch_size=4 `
            --lang=multi `
            --trust_remote_code `
            --inference_stride=128 `
            --output_file_prefix=$output_prefix `
            --inference_filter_file=$filter_file

        if ($LASTEXITCODE -ne 0) {
            $err = "[$(Get-Date)] ERROR for language: $lang (exit code $LASTEXITCODE)"
            Write-Warning $err
            Add-Content $logFile $err
            continue   # move to next language instead of stopping
        }

        $done = "[$(Get-Date)] Finished language: $lang"
        Write-Host $done
        Add-Content $logFile $done
    }
    catch {
        $err = "[$(Get-Date)] EXCEPTION for language: $lang - $_"
        Write-Error $err
        Add-Content $logFile $err
        continue
    }
}

Write-Host "All jobs completed. Press any key to exit..."
Pause