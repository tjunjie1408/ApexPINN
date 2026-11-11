locals {
  studio_app_arn = "arn:aws:sagemaker:${var.region}:${var.account_id}:app/${aws_sagemaker_domain.studio.id}/${aws_sagemaker_space.training.space_name}/JupyterLab/default"
}

# Keep portal access separate from the S3 policy needed before domain creation.
resource "aws_iam_role_policy" "studio_portal" {
  name = "apexpinn-studio-portal"
  role = aws_iam_role.studio.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "DescribePortalCompatibilityApp"
        Effect   = "Allow"
        Action   = ["sagemaker:DescribeApp"]
        Resource = "arn:aws:sagemaker:${var.region}:${var.account_id}:app/${aws_sagemaker_domain.studio.id}/${aws_sagemaker_user_profile.owner.user_profile_name}/JupyterServer/default"
      },
      {
        Sid    = "ReadThisStudioSetup"
        Effect = "Allow"
        Action = [
          "sagemaker:DescribeDomain", "sagemaker:DescribeUserProfile",
          "sagemaker:DescribeSpace", "sagemaker:DescribeApp", "sagemaker:ListTags"
        ]
        Resource = [
          aws_sagemaker_domain.studio.arn,
          aws_sagemaker_user_profile.owner.arn,
          aws_sagemaker_space.training.arn,
          local.studio_app_arn
        ]
      },
      {
        Sid      = "ListStudioInSydney"
        Effect   = "Allow"
        Action   = ["sagemaker:ListApps", "sagemaker:ListDomains", "sagemaker:ListUserProfiles", "sagemaker:ListSpaces"]
        Resource = "*"
        Condition = {
          StringEquals = { "aws:RequestedRegion" = var.region }
        }
      },
      {
        Sid      = "OpenThisUserProfile"
        Effect   = "Allow"
        Action   = ["sagemaker:CreatePresignedDomainUrl"]
        Resource = aws_sagemaker_user_profile.owner.arn
      },
      {
        Sid      = "ManageThisPrivateJupyterLabApp"
        Effect   = "Allow"
        Action   = ["sagemaker:CreateApp", "sagemaker:DeleteApp"]
        Resource = local.studio_app_arn
        Condition = {
          ArnEquals    = { "sagemaker:OwnerUserProfileArn" = aws_sagemaker_user_profile.owner.arn }
          StringEquals = { "sagemaker:SpaceSharingType" = "Private" }
        }
      },
      {
        Sid      = "ConfigureThisPrivateSpace"
        Effect   = "Allow"
        Action   = ["sagemaker:UpdateSpace"]
        Resource = aws_sagemaker_space.training.arn
        Condition = {
          ArnEquals    = { "sagemaker:OwnerUserProfileArn" = aws_sagemaker_user_profile.owner.arn }
          StringEquals = { "sagemaker:SpaceSharingType" = "Private" }
        }
      },
      {
        Sid      = "TagThisAppDuringCreation"
        Effect   = "Allow"
        Action   = ["sagemaker:AddTags"]
        Resource = local.studio_app_arn
        Condition = {
          StringEquals = { "sagemaker:TaggingAction" = "CreateApp" }
        }
      }
    ]
  })
}
