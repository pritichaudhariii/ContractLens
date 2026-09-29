output "alb_url" {
  description = "Public URL of the API"
  value       = "http://${aws_lb.app.dns_name}"
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "ecs_cluster" {
  value = aws_ecs_cluster.main.name
}

output "ecs_service" {
  value = aws_ecs_service.app.name
}

output "db_endpoint" {
  value = aws_db_instance.postgres.address
}

output "log_group" {
  value = aws_cloudwatch_log_group.app.name
}
