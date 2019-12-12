# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: imu.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import point3f_pb2 as point3f__pb2
import point4f_pb2 as point4f__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='imu.proto',
  package='ltpb',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\timu.proto\x12\x04ltpb\x1a\rpoint3f.proto\x1a\rpoint4f.proto\"\x8c\x02\n\tImuSample\x12\x11\n\tdevice_id\x18\x01 \x01(\x04\x12\x13\n\x0blocation_id\x18\x02 \x01(\r\x12\x10\n\x08raw_time\x18\x03 \x01(\x04\x12\x13\n\x0btemperature\x18\x04 \x01(\x02\x12\x1c\n\x05\x65uler\x18\x05 \x01(\x0b\x32\r.ltpb.Point3F\x12#\n\x0c\x61\x63\x63\x65leration\x18\x06 \x01(\x0b\x32\r.ltpb.Point3F\x12(\n\x11\x66ree_acceleration\x18\x07 \x01(\x0b\x32\r.ltpb.Point3F\x12 \n\tgyroscope\x18\x08 \x01(\x0b\x32\r.ltpb.Point3F\x12!\n\nquaternion\x18\t \x01(\x0b\x32\r.ltpb.Point4Fb\x06proto3'
  ,
  dependencies=[point3f__pb2.DESCRIPTOR,point4f__pb2.DESCRIPTOR,])




_IMUSAMPLE = _descriptor.Descriptor(
  name='ImuSample',
  full_name='ltpb.ImuSample',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='device_id', full_name='ltpb.ImuSample.device_id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='location_id', full_name='ltpb.ImuSample.location_id', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='raw_time', full_name='ltpb.ImuSample.raw_time', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='temperature', full_name='ltpb.ImuSample.temperature', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='euler', full_name='ltpb.ImuSample.euler', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='acceleration', full_name='ltpb.ImuSample.acceleration', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='free_acceleration', full_name='ltpb.ImuSample.free_acceleration', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gyroscope', full_name='ltpb.ImuSample.gyroscope', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='quaternion', full_name='ltpb.ImuSample.quaternion', index=8,
      number=9, type=11, cpp_type=10, label=1,
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
  serialized_start=50,
  serialized_end=318,
)

_IMUSAMPLE.fields_by_name['euler'].message_type = point3f__pb2._POINT3F
_IMUSAMPLE.fields_by_name['acceleration'].message_type = point3f__pb2._POINT3F
_IMUSAMPLE.fields_by_name['free_acceleration'].message_type = point3f__pb2._POINT3F
_IMUSAMPLE.fields_by_name['gyroscope'].message_type = point3f__pb2._POINT3F
_IMUSAMPLE.fields_by_name['quaternion'].message_type = point4f__pb2._POINT4F
DESCRIPTOR.message_types_by_name['ImuSample'] = _IMUSAMPLE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ImuSample = _reflection.GeneratedProtocolMessageType('ImuSample', (_message.Message,), {
  'DESCRIPTOR' : _IMUSAMPLE,
  '__module__' : 'imu_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.ImuSample)
  })
_sym_db.RegisterMessage(ImuSample)


# @@protoc_insertion_point(module_scope)
