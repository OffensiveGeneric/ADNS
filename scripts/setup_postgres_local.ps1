# Creates a local PostgreSQL database/user for ADNS.
# Usage: pwsh ./scripts/setup_postgres_local.ps1 -Database adns -User adns -Password adns_password
[CmdletBinding()]
param(
    [string]$Database = "adns",
    [string]$User = "adns",
    [string]$Password = "adns_password",
    [string]$Host = "127.0.0.1",
    [int]$Port = 5432,
    [string]$Superuser = "postgres",
    [string]$SuperuserPassword = $env:PGPASSWORD
)

$ErrorActionPreference = "Stop"

function Require-Command {
    param(
        [string]$Name,
        [string]$Message
    )
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw $Message
    }
}

Require-Command -Name "psql" -Message "psql not found. Install PostgreSQL and ensure it is on PATH (e.g., winget install -e --id PostgreSQL.PostgreSQL)."

if (-not $SuperuserPassword) {
    $secure = Read-Host -Prompt "Enter password for PostgreSQL superuser '$Superuser'" -AsSecureString
    $SuperuserPassword = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
}

$env:PGPASSWORD = $SuperuserPassword
$commonArgs = @("-h", $Host, "-p", $Port, "-U", $Superuser, "-d", "postgres", "-v", "ON_ERROR_STOP=1")

Write-Host "Ensuring role '$User' exists..."
& psql @commonArgs -c "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '$User') THEN CREATE ROLE $User LOGIN PASSWORD '$Password'; END IF; END $$;"

Write-Host "Ensuring database '$Database' exists (owned by '$User')..."
& psql @commonArgs -c "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_database WHERE datname = '$Database') THEN CREATE DATABASE $Database WITH OWNER $User ENCODING 'UTF8'; END IF; END $$;"

Write-Host "Granting privileges on '$Database' to '$User'..."
& psql @commonArgs -c "ALTER DATABASE $Database OWNER TO $User; GRANT ALL PRIVILEGES ON DATABASE $Database TO $User;"

Write-Host "`nDatabase ready. Add this to your .env (or export it) and restart services:"
Write-Host "SQLALCHEMY_DATABASE_URI=postgresql://$User:$Password@$Host:$Port/$Database"
Write-Host "ADNS_REDIS_URL=redis://127.0.0.1:6379/0"
