# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: imu_data.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import point3f_pb2 as point3f__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='imu_data.proto',
  package='ltpb',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x0eimu_data.proto\x12\x04ltpb\x1a\rpoint3f.proto\"\xac\x01\n\x07IMUData\x12\x13\n\x0b\x66rame_index\x18\x01 \x01(\r\x12+\n\raccelerometer\x18\x02 \x03(\x0b\x32\x14.ltpb.IMUData.Sample\x12\'\n\tgyroscope\x18\x03 \x03(\x0b\x32\x14.ltpb.IMUData.Sample\x1a\x36\n\x06Sample\x12\x0f\n\x07row_idx\x18\x01 \x01(\r\x12\x1b\n\x04\x64\x61ta\x18\x02 \x01(\x0b\x32\r.ltpb.Point3Fb\x06proto3'
  ,
  dependencies=[point3f__pb2.DESCRIPTOR,])




_IMUDATA_SAMPLE = _descriptor.Descriptor(
  name='Sample',
  full_name='ltpb.IMUData.Sample',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='row_idx', full_name='ltpb.IMUData.Sample.row_idx', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='ltpb.IMUData.Sample.data', index=1,
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
  serialized_start=158,
  serialized_end=212,
)

_IMUDATA = _descriptor.Descriptor(
  name='IMUData',
  full_name='ltpb.IMUData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='frame_index', full_name='ltpb.IMUData.frame_index', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='accelerometer', full_name='ltpb.IMUData.accelerometer', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gyroscope', full_name='ltpb.IMUData.gyroscope', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_IMUDATA_SAMPLE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=40,
  serialized_end=212,
)

_IMUDATA_SAMPLE.fields_by_name['data'].message_type = point3f__pb2._POINT3F
_IMUDATA_SAMPLE.containing_type = _IMUDATA
_IMUDATA.fields_by_name['accelerometer'].message_type = _IMUDATA_SAMPLE
_IMUDATA.fields_by_name['gyroscope'].message_type = _IMUDATA_SAMPLE
DESCRIPTOR.message_types_by_name['IMUData'] = _IMUDATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

IMUData = _reflection.GeneratedProtocolMessageType('IMUData', (_message.Message,), {

  'Sample' : _reflection.GeneratedProtocolMessageType('Sample', (_message.Message,), {
    'DESCRIPTOR' : _IMUDATA_SAMPLE,
    '__module__' : 'imu_data_pb2'
    # @@protoc_insertion_point(class_scope:ltpb.IMUData.Sample)
    })
  ,
  'DESCRIPTOR' : _IMUDATA,
  '__module__' : 'imu_data_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.IMUData)
  })
_sym_db.RegisterMessage(IMUData)
_sym_db.RegisterMessage(IMUData.Sample)


# @@protoc_insertion_point(module_scope)
