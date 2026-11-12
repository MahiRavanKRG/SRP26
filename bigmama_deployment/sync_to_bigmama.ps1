param (
    [Parameter(Mandatory=$false)]
    [string]$Username,
    [string]$HostName = "master.ismll.de",
    [string]$RemoteDir = "~/SRP26"
)

if (-not $Username) {
    $Username = Read-Host "Please enter your Bigmama / Uni Hildesheim username"
}

# Sanitize username if passed as <user@uni-hildesheim.de> or user@uni-hildesheim.de
$Username = $Username.Trim("<>").Trim()
if ($Username -match "@") {
    $Username = $Username.Split("@")[0]
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Syncing SRP26 Bigmama Deployment to $Username@$HostName" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Check if SSH connection works
Write-Host "Testing SSH reachability to $HostName ..." -ForegroundColor Yellow
$test = Test-NetConnection -ComputerName $HostName -Port 22 -WarningAction SilentlyContinue
if (-not $test.TcpTestSucceeded) {
    Write-Host "[WARNING] Could not connect to $HostName:22!" -ForegroundColor Red
    Write-Host "Please make sure you are connected to the University of Hildesheim VPN (Cisco AnyConnect / eduVPN)." -ForegroundColor Yellow
    $proceed = Read-Host "Do you want to attempt scp anyway? (y/n)"
    if ($proceed.ToLower() -ne "y") {
        Exit
    }
}

Write-Host "`n1. Creating remote directory $RemoteDir on $HostName ..." -ForegroundColor Green
ssh "$Username@$HostName" "mkdir -p $RemoteDir/data $RemoteDir/results/cluster_results $RemoteDir/logs"

Write-Host "`n2. Uploading deployment package ..." -ForegroundColor Green
scp -r "d:\uni\SRP26\bigmama_deployment\*" "$Username@${HostName}:$RemoteDir/"

Write-Host "`n3. Uploading datasets (ETTh1, ETTh2, weather, traffic, electricity) ..." -ForegroundColor Green
scp "d:\uni\SRP26\data_sets\ETTh1.csv" "$Username@${HostName}:$RemoteDir/data/"
scp "d:\uni\SRP26\data_sets\ETTh2.csv" "$Username@${HostName}:$RemoteDir/data/"
scp "d:\uni\SRP26\data_sets\weather.csv" "$Username@${HostName}:$RemoteDir/data/"
scp "d:\uni\SRP26\data_sets\traffic.csv" "$Username@${HostName}:$RemoteDir/data/"
scp "d:\uni\SRP26\data_sets\electricity.csv" "$Username@${HostName}:$RemoteDir/data/"

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " Sync Complete! To run experiments on Bigmama:" -ForegroundColor Green
Write-Host "   1. ssh $Username@$HostName" -ForegroundColor White
Write-Host "   2. cd $RemoteDir" -ForegroundColor White
Write-Host "   3. bash setup_bigmama_env.sh" -ForegroundColor White
Write-Host "   4. sbatch submit_array_job.sh" -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Cyan

