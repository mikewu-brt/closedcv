# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lt_protobuf/light_header/device_temp.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='lt_protobuf/light_header/device_temp.proto',
  package='ltpb',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n*lt_protobuf/light_header/device_temp.proto\x12\x04ltpb\"\x87\x01\n\nDeviceTemp\x12\x10\n\x08sensor_1\x18\x01 \x01(\x11\x12\x10\n\x08sensor_2\x18\x02 \x01(\x11\x12\x10\n\x08sensor_3\x18\x03 \x01(\x11\x12\x10\n\x08sensor_4\x18\x04 \x01(\x11\x12\x17\n\rflex_sensor_1\x18\x05 \x01(\x11H\x00\x42\x18\n\x16optional_flex_sensor_1b\x06proto3')
)




_DEVICETEMP = _descriptor.Descriptor(
  name='DeviceTemp',
  full_name='ltpb.DeviceTemp',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sensor_1', full_name='ltpb.DeviceTemp.sensor_1', index=0,
      number=1, type=17, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sensor_2', full_name='ltpb.DeviceTemp.sensor_2', index=1,
      number=2, type=17, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sensor_3', full_name='ltpb.DeviceTemp.sensor_3', index=2,
      number=3, type=17, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sensor_4', full_name='ltpb.DeviceTemp.sensor_4', index=3,
      number=4, type=17, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='flex_sensor_1', full_name='ltpb.DeviceTemp.flex_sensor_1', index=4,
      number=5, type=17, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
    _descriptor.OneofDescriptor(
      name='optional_flex_sensor_1', full_name='ltpb.DeviceTemp.optional_flex_sensor_1',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=53,
  serialized_end=188,
)

_DEVICETEMP.oneofs_by_name['optional_flex_sensor_1'].fields.append(
  _DEVICETEMP.fields_by_name['flex_sensor_1'])
_DEVICETEMP.fields_by_name['flex_sensor_1'].containing_oneof = _DEVICETEMP.oneofs_by_name['optional_flex_sensor_1']
DESCRIPTOR.message_types_by_name['DeviceTemp'] = _DEVICETEMP
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DeviceTemp = _reflection.GeneratedProtocolMessageType('DeviceTemp', (_message.Message,), {
  'DESCRIPTOR' : _DEVICETEMP,
  '__module__' : 'lt_protobuf.light_header.device_temp_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.DeviceTemp)
  })
_sym_db.RegisterMessage(DeviceTemp)


# @@protoc_insertion_point(module_scope)
