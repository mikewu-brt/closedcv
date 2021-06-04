# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lt_protobuf/light_header/extra_sensors/extra_sensors.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from lt_protobuf.light_header.extra_sensors import gps_data_pb2 as lt__protobuf_dot_light__header_dot_extra__sensors_dot_gps__data__pb2
from lt_protobuf.light_header.extra_sensors import lidar_data_pb2 as lt__protobuf_dot_light__header_dot_extra__sensors_dot_lidar__data__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='lt_protobuf/light_header/extra_sensors/extra_sensors.proto',
  package='ltpb',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n:lt_protobuf/light_header/extra_sensors/extra_sensors.proto\x12\x04ltpb\x1a\x35lt_protobuf/light_header/extra_sensors/gps_data.proto\x1a\x37lt_protobuf/light_header/extra_sensors/lidar_data.proto\"\xf2\x01\n\x0c\x45xtraSensors\x12$\n\x0bgps_hw_info\x18\x01 \x03(\x0b\x32\x0f.ltpb.GPSHwInfo\x12\x1f\n\x08gps_data\x18\x02 \x03(\x0b\x32\r.ltpb.GPSData\x12(\n\rlidar_hw_info\x18\x03 \x03(\x0b\x32\x11.ltpb.LidarHwInfo\x12\x31\n\x11lidar_calibration\x18\x04 \x03(\x0b\x32\x16.ltpb.LidarCalibration\x12>\n\x18lidar_camera_calibration\x18\x05 \x03(\x0b\x32\x1c.ltpb.LidarCameraCalibrationb\x06proto3'
  ,
  dependencies=[lt__protobuf_dot_light__header_dot_extra__sensors_dot_gps__data__pb2.DESCRIPTOR,lt__protobuf_dot_light__header_dot_extra__sensors_dot_lidar__data__pb2.DESCRIPTOR,])




_EXTRASENSORS = _descriptor.Descriptor(
  name='ExtraSensors',
  full_name='ltpb.ExtraSensors',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='gps_hw_info', full_name='ltpb.ExtraSensors.gps_hw_info', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gps_data', full_name='ltpb.ExtraSensors.gps_data', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lidar_hw_info', full_name='ltpb.ExtraSensors.lidar_hw_info', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lidar_calibration', full_name='ltpb.ExtraSensors.lidar_calibration', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lidar_camera_calibration', full_name='ltpb.ExtraSensors.lidar_camera_calibration', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=181,
  serialized_end=423,
)

_EXTRASENSORS.fields_by_name['gps_hw_info'].message_type = lt__protobuf_dot_light__header_dot_extra__sensors_dot_gps__data__pb2._GPSHWINFO
_EXTRASENSORS.fields_by_name['gps_data'].message_type = lt__protobuf_dot_light__header_dot_extra__sensors_dot_gps__data__pb2._GPSDATA
_EXTRASENSORS.fields_by_name['lidar_hw_info'].message_type = lt__protobuf_dot_light__header_dot_extra__sensors_dot_lidar__data__pb2._LIDARHWINFO
_EXTRASENSORS.fields_by_name['lidar_calibration'].message_type = lt__protobuf_dot_light__header_dot_extra__sensors_dot_lidar__data__pb2._LIDARCALIBRATION
_EXTRASENSORS.fields_by_name['lidar_camera_calibration'].message_type = lt__protobuf_dot_light__header_dot_extra__sensors_dot_lidar__data__pb2._LIDARCAMERACALIBRATION
DESCRIPTOR.message_types_by_name['ExtraSensors'] = _EXTRASENSORS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ExtraSensors = _reflection.GeneratedProtocolMessageType('ExtraSensors', (_message.Message,), {
  'DESCRIPTOR' : _EXTRASENSORS,
  '__module__' : 'lt_protobuf.light_header.extra_sensors.extra_sensors_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.ExtraSensors)
  })
_sym_db.RegisterMessage(ExtraSensors)


# @@protoc_insertion_point(module_scope)