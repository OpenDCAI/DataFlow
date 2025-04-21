<#
    .SYNOPSIS
    Create a new role definition.

    .DESCRIPTION
    This API registers a new role for an organization. Organization roles are unique within the organization,
    and can be created by administrators. We recommend creating and managing roles at the top organization
    level instead of in a sub-organization.

    .INPUTS
    An organization resource object

    .OUTPUTS
    An role resource object

    .PARAMETER Org
    An organization resource object

    .PARAMETER Name
    The role name

    .PARAMETER Description
    The role description

    .EXAMPLE
    $role | Add-Permissions @("PATIENT.READ", "PATIENT.WRITE")

    .LINK
    https://www.hsdp.io/documentation/identity-and-access-management-iam/api-documents/resource-reference-api/role-api#/Role%20Management/Create%20Role

    .NOTES
    POST: /authorize/identity/Role/{id}}/$assign-permission v1
#>
function Add-Role {

    [CmdletBinding()]
    [OutputType([PSObject])]
    param(
        [Parameter(Mandatory, Position = 0, ValueFromPipeline)]
        [ValidateNotNullOrEmpty()]
        [PSObject]$Org,

        [Parameter(Mandatory, Position = 1)]
        [ValidateNotNullOrEmpty()]
        [String]$Name,

        [Parameter(Mandatory, Position = 2)]
        [ValidateNotNullOrEmpty()]
        [String]$Description
    )

    begin {
        Write-Verbose "[$($MyInvocation.MyCommand.Name)] Function started"
    }

    process {
        Write-Debug "[$($MyInvocation.MyCommand.Name)] PSBoundParameters: $($PSBoundParameters | Out-String)"

        $body = @{
            "name"= $Name;
            "description"= $Description;
            "managingOrganization"= $Org.id;
        }

        $response  = (Invoke-ApiRequest -Path "/authorize/identity/Role" -Version 1 -Method Post -Body $body -ValidStatusCodes @(201))
        Write-Output @($response)
    }

    end {
        Write-Verbose "[$($MyInvocation.MyCommand.Name)] Complete"
    }
}