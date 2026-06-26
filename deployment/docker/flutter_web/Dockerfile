FROM nginx:alpine

# Placeholder for future Flutter web build artifacts
COPY deployment/docker/flutter_web/default.conf /etc/nginx/conf.d/default.conf
RUN mkdir -p /usr/share/nginx/html && echo "MindGraph++ Web" > /usr/share/nginx/html/index.html
