FROM nginx:1.27-alpine

COPY deployment/docker/nginx/nginx.conf /etc/nginx/nginx.conf
COPY deployment/docker/nginx/conf.d/ /etc/nginx/conf.d/
COPY deployment/docker/nginx/certs/ /etc/nginx/certs/

EXPOSE 80 443

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/health || exit 1
