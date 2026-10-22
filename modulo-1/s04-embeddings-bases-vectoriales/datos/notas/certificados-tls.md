# Certificados TLS

Los certificados TLS permiten conexiones seguras (https) a los servicios internos y a los portales de clientes. Cada certificado tiene una fecha de expiración; cuando vence, los navegadores muestran una advertencia y los balanceadores pueden rechazar las conexiones, lo que deja el servicio inaccesible.

La renovación es responsabilidad del equipo de plataforma y debe iniciarse 15 días antes de la fecha de expiración. Existe un calendario compartido con todos los vencimientos y una alerta automática a los 30, 15 y 7 días.

Después de renovar, hay que instalar la cadena completa (certificado, intermedios) y reiniciar el balanceador si es necesario. Verifica siempre con una prueba externa que la cadena sea correcta.
