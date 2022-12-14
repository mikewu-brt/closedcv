# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lt_protobuf/light_header/sensor_type.proto

from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='lt_protobuf/light_header/sensor_type.proto',
  package='ltpb',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n*lt_protobuf/light_header/sensor_type.proto\x12\x04ltpb*\xfa\x01\n\nSensorType\x12\x12\n\x0eSENSOR_UNKNOWN\x10\x00\x12\x10\n\x0cSENSOR_AR835\x10\x01\x12\x11\n\rSENSOR_AR1335\x10\x02\x12\x16\n\x12SENSOR_AR1335_MONO\x10\x03\x12\x11\n\rSENSOR_IMX386\x10\x04\x12\x16\n\x12SENSOR_IMX386_MONO\x10\x05\x12\x11\n\rSENSOR_IMX350\x10\x06\x12\x11\n\rSENSOR_IMX398\x10\x07\x12\x11\n\rSENSOR_IMX586\x10\x08\x12\x11\n\rSENSOR_IMX420\x10\t\x12\x11\n\rSENSOR_IMX428\x10\n\x12\x11\n\rSENSOR_IMX265\x10\x0b\x62\x06proto3'
)

_SENSORTYPE = _descriptor.EnumDescriptor(
  name='SensorType',
  full_name='ltpb.SensorType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='SENSOR_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SENSOR_AR835', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SENSOR_AR1335', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SENSOR_AR1335_MONO', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SENSOR_IMX386', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SENSOR_IMX386_MONO', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SENSOR_IMX350', index=6, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SENSOR_IMX398', index=7, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SENSOR_IMX586', index=8, number=8,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SENSOR_IMX420', index=9, number=9,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SENSOR_IMX428', index=10, number=10,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SENSOR_IMX265', index=11, number=11,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=53,
  serialized_end=303,
)
_sym_db.RegisterEnumDescriptor(_SENSORTYPE)

SensorType = enum_type_wrapper.EnumTypeWrapper(_SENSORTYPE)
SENSOR_UNKNOWN = 0
SENSOR_AR835 = 1
SENSOR_AR1335 = 2
SENSOR_AR1335_MONO = 3
SENSOR_IMX386 = 4
SENSOR_IMX386_MONO = 5
SENSOR_IMX350 = 6
SENSOR_IMX398 = 7
SENSOR_IMX586 = 8
SENSOR_IMX420 = 9
SENSOR_IMX428 = 10
SENSOR_IMX265 = 11


DESCRIPTOR.enum_types_by_name['SensorType'] = _SENSORTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)


# @@protoc_insertion_point(module_scope)
