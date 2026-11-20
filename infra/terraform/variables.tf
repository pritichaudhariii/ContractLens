variable "region" {
  type    = string
  default = "us-east-1"
}

variable "environment" {
  type    = string
  default = "staging"
}

variable "image_tag" {
  description = "ECR image tag to deploy (the deploy workflow sets this to the git SHA)"
  type        = string
  default     = "latest"
}

variable "container_port" {
  type    = number
  default = 8000
}

variable "task_cpu" {
  type    = number
  default = 512
}

variable "task_memory" {
  type    = number
  default = 1024
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "db_instance_class" {
  type    = string
  default = "db.t4g.micro"
}

variable "db_password" {
  type      = string
  sensitive = true
}

variable "azure_foundry_endpoint" {
  description = "Azure AI Foundry endpoint, e.g. https://<resource>.openai.azure.com"
  type        = string
}

variable "azure_foundry_api_key" {
  type      = string
  sensitive = true
}

variable "azure_foundry_api_style" {
  description = "azure_openai (deployment names) or inference (Foundry model inference API)"
  type        = string
  default     = "azure_openai"
}

variable "azure_foundry_chat_deployment" {
  type    = string
  default = "gpt-4o-mini"
}

variable "azure_foundry_embedding_deployment" {
  type    = string
  default = "text-embedding-3-small"
}

variable "azure_foundry_embedding_dimension" {
  type    = number
  default = 1536
}

variable "voyage_api_key" {
  type      = string
  sensitive = true
  default   = ""
}

variable "api_key" {
  description = "Value clients must send as X-API-Key; empty disables auth"
  type        = string
  sensitive   = true
  default     = ""
}


