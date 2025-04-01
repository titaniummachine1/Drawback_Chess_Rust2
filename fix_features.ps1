$content = Get-Content -Path src/engine.rs -Raw
$newContent = $content -replace 'salewskiChessDebug', 'drawbackChessDebug'
$newContent | Set-Content -Path src/engine.rs -NoNewline

Write-Host "Replaced all occurrences of salewskiChessDebug with drawbackChessDebug in engine.rs" 