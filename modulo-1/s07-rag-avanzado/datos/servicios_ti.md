# Catálogo de servicios de TI de Andes Fibra

Este catálogo es material ficticio creado para el curso. Describe los servicios de TI de Andes Fibra, un proveedor imaginario de internet, y cómo se relacionan entre sí. Lo consulta la mesa de ayuda para atender incidentes. Cada sección describe un servicio, de qué depende y qué equipo lo administra. Las dependencias se describen solo desde el punto de vista de cada servicio.

## Centro de datos Norte

Es el centro de datos principal de Andes Fibra. Aloja los servidores de producción, el almacenamiento y la red troncal. Lo administra el Equipo Infraestructura.

## Centro de datos Sur

Es el centro de datos secundario, usado para copias de seguridad. No atiende tráfico de clientes. Lo administra el Equipo Infraestructura.

## Red troncal

Es la red de alta capacidad que conecta los servidores entre sí y con los clientes. Se ejecuta en el Centro de datos Norte. Lo administra el Equipo Redes.

## Base de datos de clientes

Guarda los contratos, los datos de facturación y el historial de cada cliente. Es una base PostgreSQL con réplica de lectura. Para funcionar depende de la Red troncal. Se ejecuta en el Centro de datos Norte. Lo administra el Equipo Datos.

## Almacenamiento de respaldos

Guarda las copias de seguridad nocturnas de las bases de datos, con retención de 30 días. Para funcionar depende de la Red troncal. Se ejecuta en el Centro de datos Sur. Lo administra el Equipo Datos.

## Directorio de usuarios

Es el directorio LDAP con las cuentas del personal y de los clientes empresariales. Se ejecuta en el Centro de datos Norte. Lo administra el Equipo Seguridad.

## Servicio de autenticación

Valida el usuario y la clave, y entrega el inicio de sesión único (SSO) a las demás aplicaciones. Para funcionar depende del Directorio de usuarios y la Base de datos de clientes. Lo administra el Equipo Seguridad.

## Cola de mensajes

Recibe y reparte los mensajes que las aplicaciones se envían entre sí, para que no se pierdan si un servicio está ocupado. Se ejecuta en el Centro de datos Norte. Lo administra el Equipo Plataforma.

## Proveedor SMS externo

Es la empresa que entrega los mensajes de texto a los teléfonos de los clientes. No es de Andes Fibra y se contrata por volumen.

## Servicio de notificaciones

Envía correos y mensajes de texto: avisos de factura, códigos de verificación y alertas de falla. Para funcionar depende de la Cola de mensajes y el Proveedor SMS externo. Lo administra el Equipo Plataforma.

## API de facturación

Calcula las facturas, aplica descuentos y recargos, y expone esos datos a las aplicaciones. Para funcionar depende de la Base de datos de clientes y el Servicio de autenticación. Lo administra el Equipo Pagos.

## Pasarela de pagos

Procesa los pagos con tarjeta y las transferencias, y registra el resultado de cada cobro. Para funcionar depende de la API de facturación. Lo administra el Equipo Pagos.

## Portal Mi Andes

Es el sitio web donde los clientes ven su factura, pagan, piden prórrogas y solicitan cancelaciones. Para funcionar depende del Servicio de autenticación, la API de facturación, la Pasarela de pagos y el Servicio de notificaciones. Lo administra el Equipo Web.

## App móvil

Es la aplicación para teléfonos con las funciones básicas del portal: ver la factura y reportar fallas. Para funcionar depende del Servicio de autenticación y la API de facturación. Lo administra el Equipo Web.

## Sistema de tickets

Registra cada caso de soporte, su estado y su historial. Lo usan la mesa de ayuda y los técnicos de campo. Para funcionar depende de la Base de datos de clientes, el Servicio de notificaciones y el Servicio de autenticación. Lo administra el Equipo Soporte TI.

## Chat de soporte

Es el chat del portal con el que el cliente habla con la mesa de ayuda; cada conversación se guarda como un ticket. Para funcionar depende del Sistema de tickets y el Servicio de autenticación. Lo administra el Equipo Soporte TI.
