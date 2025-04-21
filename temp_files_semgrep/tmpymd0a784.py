<#
This Sample Code is provided for the purpose of illustration only
and is not intended to be used in a production environment.  THIS
SAMPLE CODE AND ANY RELATED INFORMATION ARE PROVIDED "AS IS" WITHOUT
WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS
FOR A PARTICULAR PURPOSE.  We grant You a nonexclusive, royalty-free
right to use and modify the Sample Code and to reproduce and distribute
the object code form of the Sample Code, provided that You agree:
(i) to not use Our name, logo, or trademarks to market Your software
product in which the Sample Code is embedded; (ii) to include a valid
copyright notice on Your software product in which the Sample Code is
embedded; and (iii) to indemnify, hold harmless, and defend Us and
Our suppliers from and against any claims or lawsuits, including
attorneys' fees, that arise or result from the use or distribution
of the Sample Code.
#>
#requires -Version 5 -Modules AutomatedLab -RunAsAdministrator 
trap {
    Write-Host "Stopping Transcript ..."
    Stop-Transcript
    $VerbosePreference = $PreviousVerbosePreference
    $ErrorActionPreference = $PreviousErrorActionPreference
    [console]::beep(3000, 750)
    Send-ALNotification -Activity 'Lab started' -Message ('Lab deployment failed !') -Provider (Get-LabConfigurationItem -Name Notifications.SubscribedProviders)
    break
} 
Import-Module -Name AutomatedLab -Verbose
try {while (Stop-Transcript) {}} catch {}
Clear-Host

$PreviousVerbosePreference = $VerbosePreference
$VerbosePreference = 'SilentlyContinue'
$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$CurrentScript = $MyInvocation.MyCommand.Path
#Getting the current directory (where this script file resides)
$CurrentDir = Split-Path -Path $CurrentScript -Parent
$TranscriptFile = $CurrentScript -replace ".ps1$", "$("_{0}.txt" -f (Get-Date -Format 'yyyyMMddHHmmss'))"
Start-Transcript -Path $TranscriptFile -IncludeInvocationHeader

#region Global variables definition
$Logon = 'Administrator'
$ClearTextPassword = 'P@ssw0rd'
$SecurePassword = ConvertTo-SecureString -String $ClearTextPassword -AsPlainText -Force
$NetBiosDomainName = 'CONTOSO'
$FQDNDomainName = 'contoso.com'

$NetworkID = '10.0.0.0/16' 
$DC01IPv4Address = '10.0.0.1'
$IIS01IPv4Address = '10.0.0.11'
$IIS02IPv4Address = '10.0.0.12'
$WAC01IPv4Address = '10.0.0.21'

$WACDownloadURI = "http://aka.ms/WACDownload"

$LabName = 'IISWAC'
#endregion

#Cleaning previously existing lab
if ($LabName -in (Get-Lab -List)) {
    Remove-Lab -Name $LabName -Confirm:$false -ErrorAction SilentlyContinue
}

#create an empty lab template and define where the lab XML files and the VMs will be stored
New-LabDefinition -Name $LabName -DefaultVirtualizationEngine HyperV

#make the network definition
Add-LabVirtualNetworkDefinition -Name $LabName -HyperVProperties @{
    SwitchType = 'Internal'
} -AddressSpace $NetworkID
Add-LabVirtualNetworkDefinition -Name 'Default Switch' -HyperVProperties @{ SwitchType = 'External'; AdapterName = 'Wi-Fi' }


#and the domain definition with the domain admin account
Add-LabDomainDefinition -Name $FQDNDomainName -AdminUser $Logon -AdminPassword $ClearTextPassword

#these credentials are used for connecting to the machines. As this is a lab we use clear-text passwords
Set-LabInstallationCredential -Username $Logon -Password $ClearTextPassword

#defining default parameter values, as these ones are the same for all the machines
$PSDefaultParameterValues = @{
    'Add-LabMachineDefinition:Network'         = $LabName
    'Add-LabMachineDefinition:DomainName'      = $FQDNDomainName
    'Add-LabMachineDefinition:MinMemory'       = 1GB
    'Add-LabMachineDefinition:MaxMemory'       = 3GB
    'Add-LabMachineDefinition:Memory'          = 2GB
    'Add-LabMachineDefinition:OperatingSystem' = 'Windows Server 2022 Datacenter (Desktop Experience)'
    #'Add-LabMachineDefinition:Processors'      = 4
}

#region server definitions
#Domain controller + Certificate Authority
Add-LabMachineDefinition -Name DC01 -Roles RootDC, CARoot -IpAddress $DC01IPv4Address
#IIS front-end server
$IIS01NetAdapter = @()
$IIS01NetAdapter += New-LabNetworkAdapterDefinition -VirtualSwitch $LabName -Ipv4Address $IIS01IPv4Address
$IIS01NetAdapter += New-LabNetworkAdapterDefinition -VirtualSwitch 'Default Switch' -UseDhcp
Add-LabMachineDefinition -Name IIS01 -NetworkAdapter $IIS01NetAdapter

$IIS02NetAdapter = @()
$IIS02NetAdapter += New-LabNetworkAdapterDefinition -VirtualSwitch $LabName -Ipv4Address $IIS02IPv4Address
$IIS02NetAdapter += New-LabNetworkAdapterDefinition -VirtualSwitch 'Default Switch' -UseDhcp
Add-LabMachineDefinition -Name IIS02 -NetworkAdapter $IIS02NetAdapter
#Windows Admin Center Management Server
$WAC01NetAdapter = @()
$WAC01NetAdapter += New-LabNetworkAdapterDefinition -VirtualSwitch $LabName -Ipv4Address $WAC01IPv4Address
$WAC01NetAdapter += New-LabNetworkAdapterDefinition -VirtualSwitch 'Default Switch' -UseDhcp
Add-LabMachineDefinition -Name WAC01 -NetworkAdapter $WAC01NetAdapter
#endregion

#Installing servers
Install-Lab
Checkpoint-LabVM -SnapshotName AllLabVMInstall -All
#Restore-LabVMSnapshot -SnapshotName 'AllLabVMInstall' -All -Verbose

$AllLabVMs = Get-LabVM -All
$IISServers = $AllLabVMs | Where-Object -FilterScript { $_.Name -like "IIS*" }
#region Installing Required Windows Features
$Job = @()
$Job += Install-LabWindowsFeature -FeatureName Telnet-Client -ComputerName $AllLabVMs -IncludeManagementTools -AsJob -PassThru
$Job += Install-LabWindowsFeature -FeatureName Web-Server -ComputerName $IISServers -IncludeManagementTools -AsJob -PassThru
#endregion


Invoke-LabCommand -ActivityName "Disabling IE ESC" -ComputerName $AllLabVMs -ScriptBlock {
    #Disabling IE ESC
    $AdminKey = "HKLM:\SOFTWARE\Microsoft\Active Setup\Installed Components\{A509B1A7-37EF-4b3f-8CFC-4F3A74704073}"
    $UserKey = "HKLM:\SOFTWARE\Microsoft\Active Setup\Installed Components\{A509B1A8-37EF-4b3f-8CFC-4F3A74704073}"
    Set-ItemProperty -Path $AdminKey -Name "IsInstalled" -Value 0
    Set-ItemProperty -Path $UserKey -Name "IsInstalled" -Value 0
    Stop-Process -Name Explorer
    #Write-Host "IE Enhanced Security Configuration (ESC) has been disabled." -ForegroundColor Green

    #Setting the Keyboard to French
    Set-WinUserLanguageList -LanguageList "fr-FR" -Force

    #Renaming the main NIC adapter to Corp
    Rename-NetAdapter -Name "$using:labName 0" -NewName 'Corp' -PassThru -ErrorAction SilentlyContinue
    Rename-NetAdapter -Name "Ethernet" -NewName 'Corp' -PassThru -ErrorAction SilentlyContinue
    Rename-NetAdapter -Name "Default Switch 0" -NewName 'Internet' -PassThru -ErrorAction SilentlyContinue
}

#Installing and setting up DNS
Invoke-LabCommand -ActivityName 'DNS, AD Setup & GPO Settings on DC' -ComputerName DC01 -ScriptBlock {
    #region DNS management
    #Reverse lookup zone creation
    Add-DnsServerPrimaryZone -NetworkID $using:NetworkID -ReplicationScope 'Forest' 
    #endregion

    $DefaultNamingContext = (Get-ADRootDSE).defaultNamingContext
    #region Edge Settings
    $GPO = New-GPO -Name "Edge Settings" | New-GPLink -Target $DefaultNamingContext
    # https://devblogs.microsoft.com/powershell-community/how-to-change-the-start-page-for-the-edge-browser/
    Set-GPRegistryValue -Name $GPO.DisplayName -Key 'HKLM\Software\Policies\Microsoft\Edge' -ValueName "RestoreOnStartup" -Type ([Microsoft.Win32.RegistryValueKind]::Dword) -Value 4

    #Bonus : To open the .net core website on all machines
    $StartPage = "https://WAC01"
    Set-GPRegistryValue -Name $GPO.DisplayName -Key 'HKLM\Software\Policies\Microsoft\Edge\RestoreOnStartupURLs' -ValueName 0 -Type ([Microsoft.Win32.RegistryValueKind]::String) -Value "$StartPage"
    #https://admx.help/?Category=Windows_10_2016&Policy=Microsoft.Policies.MicrosoftEdge::PreventFirstRunPage
    Set-GPRegistryValue -Name $GPO.DisplayName -Key 'HKLM\Software\Policies\Microsoft\MicrosoftEdge\Main' -ValueName "PreventFirstRunPage" -Type ([Microsoft.Win32.RegistryValueKind]::Dword) -Value 1

    #Hide the First-run experience and splash screen on Edge : https://docs.microsoft.com/en-us/deployedge/microsoft-edge-policies#hidefirstrunexperience
    #https://admx.help/?Category=EdgeChromium&Policy=Microsoft.Policies.Edge::HideFirstRunExperience
    Set-GPRegistryValue -Name $GPO.DisplayName -Key 'HKLM\SOFTWARE\Microsoft\Edge' -ValueName "HideFirstRunExperience " -Type ([Microsoft.Win32.RegistryValueKind]::Dword) -Value 1
    #endregion

    #region WireShark : (Pre)-Master-Secret Log Filename
    $GPO = New-GPO -Name "(Pre)-Master-Secret Log Filename" | New-GPLink -Target $DefaultNamingContext
    #For decrypting SSL traffic via network tools : https://support.f5.com/csp/article/K50557518
    $SSLKeysFile = '%USERPROFILE%\AppData\Local\WireShark\ssl-keys.log'
    Set-GPRegistryValue -Name $GPO.DisplayName -Key 'HKCU\Environment' -ValueName "SSLKEYLOGFILE" -Type ([Microsoft.Win32.RegistryValueKind]::ExpandString) -Value $SSLKeysFile
    #endregion
}

#region Certification Authority : Creation and SSL Certificate Generation
#Get the CA
$CertificationAuthority = Get-LabIssuingCA
#Generating a new template for SSL Web Server certificate
New-LabCATemplate -TemplateName WebServerSSL -DisplayName 'Web Server SSL' -SourceTemplateName WebServer -ApplicationPolicy 'Server Authentication' -EnrollmentFlags Autoenrollment -PrivateKeyFlags AllowKeyExport -Version 2 -SamAccountName 'Domain Computers' -ComputerName $CertificationAuthority -ErrorAction Stop
$WAC01WebServerSSLCert = Request-LabCertificate -Subject "CN=WAC01.$FQDNDomainName" -SAN "WAC01.$FQDNDomainName", "WAC01" -TemplateName WebServerSSL -ComputerName WAC01 -PassThru -ErrorAction Stop
#region

#region Downloading and Installing Windows Admin Center Installer for hosting the web app
$WACDownload = Get-LabInternetFile -Uri $WACDownloadURI -Path $labSources\SoftwarePackages -FileName WindowsAdminCenter.msi -PassThru -Force
#Self-Signed Certificate
#$Job += Install-LabSoftwarePackage -ComputerName WAC01 -Path $WACDownload.FullName -CommandLine "/qn /L*V $env:SystemDrive\WindowsAdminCenter-Install.log ENABLE_CHK_REDIRECT_PORT_80=1 SME_AUTO_UPDATE=1 SME_PORT=443 SSL_CERTIFICATE_OPTION=generate MS_UPDATE_OPT_IN='yes' SET_TRUSTED_HOSTS=`"`"" -AsJob -PassThru
#Internal PKI Certificate, Redirection HTTP/80=>443, Enabling autoupdate, No trusted Hosts 
$Job += Install-LabSoftwarePackage -ComputerName WAC01 -Path $WACDownload.FullName -CommandLine "/qn /L*V $env:SystemDrive\WindowsAdminCenter-Install.log ENABLE_CHK_REDIRECT_PORT_80=1 SME_AUTO_UPDATE=1 SME_PORT=443 SME_THUMBPRINT=$($WAC01WebServerSSLCert.Thumbprint) SSL_CERTIFICATE_OPTION=installed MS_UPDATE_OPT_IN='yes' SET_TRUSTED_HOSTS=`"`"" -Verbose -AsJob -PassThru 
#endregion 

#Waiting for background jobs
$Job | Wait-Job | Out-Null

Invoke-LabCommand -ActivityName 'Disabling Windows Update service' -ComputerName $AllLabVMs -ScriptBlock {
    Stop-Service WUAUSERV -PassThru | Set-Service -StartupType Disabled
} 

Invoke-LabCommand -ActivityName 'Adding WAC Connections' -ComputerName WAC01 -ScriptBlock {
    # WAC URI
    $WACURI = "https://$env:COMPUTERNAME/"

    #FROM https://learn.microsoft.com/en-us/windows-server/manage/windows-admin-center/configure/use-powershell
    # Load the module
    Import-Module "$env:ProgramFiles\Windows Admin Center\PowerShell\Modules\ConnectionTools"
    # Available cmdlets: Export-Connection, Import-Connection

    $WACConnectionCSVFile = Join-Path -Path $env:SystemDrive -ChildPath "WAC-connections.csv"

    $WACConnection = $IISServers | ForEach-Object {
        [PSCustomObject] @{
            name    = $_.FQDN
            type    = "msft.sme.connection-type.server"
            tags    = @($FQDNDomainName ,"HyperV","IIS","WS2022") -join ('|')
            groupId = "global"
        }
    }
    $WACConnection | Export-Csv $WACConnectionCSVFile -NoTypeInformation
    Import-Connection $WACURI -fileName $WACConnectionCSVFile
    Remove-Item -Path $WACConnectionCSVFile -Force
} -Variable (Get-Variable -Name IISServers, FQDNDomainName)   

Invoke-LabCommand -ActivityName 'Updating & Installing WAC Extensions' -ComputerName WAC01 -ScriptBlock {
    # WAC URI
    $WACURI = "https://$env:COMPUTERNAME/"

    #FROM https://learn.microsoft.com/en-us/windows-server/manage/windows-admin-center/configure/use-powershell
    # Load the module
    Import-Module "$env:ProgramFiles\Windows Admin Center\PowerShell\Modules\ExtensionTools"
    # Available cmdlets: Get-Extension, Update-Extension, Install-Extension ...

    <#
    #FROM https://www.altf4-formation.fr/windows-admin-center-installer-des-extensions-en-powershell
    $MSFTExtensionsToInstall = 'msft.iis.iis-management', 'msft.sme.active-directory'
    #>

    #Updating Installed Extensions
    $InstalledExtensions = Get-Extension -GatewayEndpoint $WACURI | Where-Object -FilterScript { $_.status -eq 'Installed' } 
    $InstalledExtensions | ForEach-Object -Process { Write-Host "Updating $($_.title) [$($_.id)]"; Update-Extension -GatewayEndpoint $WACURI -ExtensionId $_.id -Verbose}

    #Installing all MSFT WAC entensions (not already installed)
    $MSFTExtensionsToInstall = Get-Extension -GatewayEndpoint $WACURI | Where-Object -FilterScript { $_.id -match "^msft|^microsoft" }
    $MSFTExtensionsToInstall | Where-Object -FilterScript {$_.id -notin $InstalledExtensions.id} | ForEach-Object -Process { Write-Host "Installing $($_.title) [$($_.id)]"; Install-Extension -GatewayEndpoint $WACURI -ExtensionId $_.id -Verbose}
}

#Waiting for background jobs
$Job | Wait-Job | Out-Null

#Restart is needed after the WAC installation for securing the communications
Restart-LabVM -ComputerName WAC01 -Wait

Show-LabDeploymentSummary -Detailed
Checkpoint-LabVM -SnapshotName 'FullInstall' -All

$VerbosePreference = $PreviousVerbosePreference
$ErrorActionPreference = $PreviousErrorActionPreference
#Restore-LabVMSnapshot -SnapshotName 'FullInstall' -All -Verbose

Stop-Transcript