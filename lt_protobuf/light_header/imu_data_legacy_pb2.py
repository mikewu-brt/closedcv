# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lt_protobuf/light_header/imu_data_legacy.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from lt_protobuf.common import point3f_pb2 as lt__protobuf_dot_common_dot_point3f__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='lt_protobuf/light_header/imu_data_legacy.proto',
  package='ltpb',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n.lt_protobuf/light_header/imu_data_legacy.proto\x12\x04ltpb\x1a lt_protobuf/common/point3f.proto\"\xbe\x01\n\rIMUDataLegacy\x12\x13\n\x0b\x66rame_index\x18\x01 \x01(\r\x12\x31\n\raccelerometer\x18\x02 \x03(\x0b\x32\x1a.ltpb.IMUDataLegacy.Sample\x12-\n\tgyroscope\x18\x03 \x03(\x0b\x32\x1a.ltpb.IMUDataLegacy.Sample\x1a\x36\n\x06Sample\x12\x0f\n\x07row_idx\x18\x01 \x01(\r\x12\x1b\n\x04\x64\x61ta\x18\x02 \x01(\x0b\x32\r.ltpb.Point3Fb\x06proto3'
  ,
  dependencies=[lt__protobuf_dot_common_dot_point3f__pb2.DESCRIPTOR,])




_IMUDATALEGACY_SAMPLE = _descriptor.Descriptor(
  name='Sample',
  full_name='ltpb.IMUDataLegacy.Sample',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='row_idx', full_name='ltpb.IMUDataLegacy.Sample.row_idx', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='ltpb.IMUDataLegacy.Sample.data', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  serialized_start=227,
  serialized_end=281,
)

_IMUDATALEGACY = _descriptor.Descriptor(
  name='IMUDataLegacy',
  full_name='ltpb.IMUDataLegacy',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='frame_index', full_name='ltpb.IMUDataLegacy.frame_index', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='accelerometer', full_name='ltpb.IMUDataLegacy.accelerometer', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gyroscope', full_name='ltpb.IMUDataLegacy.gyroscope', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_IMUDATALEGACY_SAMPLE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=91,
  serialized_end=281,
)

_IMUDATALEGACY_SAMPLE.fields_by_name['data'].message_type = lt__protobuf_dot_common_dot_point3f__pb2._POINT3F
_IMUDATALEGACY_SAMPLE.containing_type = _IMUDATALEGACY
_IMUDATALEGACY.fields_by_name['accelerometer'].message_type = _IMUDATALEGACY_SAMPLE
_IMUDATALEGACY.fields_by_name['gyroscope'].message_type = _IMUDATALEGACY_SAMPLE
DESCRIPTOR.message_types_by_name['IMUDataLegacy'] = _IMUDATALEGACY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

IMUDataLegacy = _reflection.GeneratedProtocolMessageType('IMUDataLegacy', (_message.Message,), {

  'Sample' : _reflection.GeneratedProtocolMessageType('Sample', (_message.Message,), {
    'DESCRIPTOR' : _IMUDATALEGACY_SAMPLE,
    '__module__' : 'lt_protobuf.light_header.imu_data_legacy_pb2'
    # @@protoc_insertion_point(class_scope:ltpb.IMUDataLegacy.Sample)
    })
  ,
  'DESCRIPTOR' : _IMUDATALEGACY,
  '__module__' : 'lt_protobuf.light_header.imu_data_legacy_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.IMUDataLegacy)
  })
_sym_db.RegisterMessage(IMUDataLegacy)
_sym_db.RegisterMessage(IMUDataLegacy.Sample)


# @@protoc_insertion_point(module_scope)
