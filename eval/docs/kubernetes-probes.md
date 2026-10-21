# Kubernetes probes

The kubelet uses probes to decide whether a container is healthy. A liveness probe detects containers that are stuck. When the liveness probe fails, the kubelet restarts the container.

A readiness probe tells Kubernetes whether a pod can receive traffic. Pods that fail the readiness probe are removed from Service endpoints but are not restarted.

A startup probe protects slow starting applications. Liveness and readiness checks are disabled until the startup probe succeeds. Probes can run an HTTP GET request, open a TCP socket, or execute a command inside the container.
