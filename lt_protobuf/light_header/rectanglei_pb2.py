# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lt_protobuf/light_header/rectanglei.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='lt_protobuf/light_header/rectanglei.proto',
  package='ltpb',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n)lt_protobuf/light_header/rectanglei.proto\x12\x04ltpb\"A\n\nRectangleI\x12\t\n\x01x\x18\x01 \x01(\x05\x12\t\n\x01y\x18\x02 \x01(\x05\x12\r\n\x05width\x18\x03 \x01(\x05\x12\x0e\n\x06height\x18\x04 \x01(\x05\x62\x06proto3'
)




_RECTANGLEI = _descriptor.Descriptor(
  name='RectangleI',
  full_name='ltpb.RectangleI',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='ltpb.RectangleI.x', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='ltpb.RectangleI.y', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='ltpb.RectangleI.width', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='ltpb.RectangleI.height', index=3,
      number=4, type=5, cpp_type=1, label=1,
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
  ],
  serialized_start=51,
  serialized_end=116,
)

DESCRIPTOR.message_types_by_name['RectangleI'] = _RECTANGLEI
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RectangleI = _reflection.GeneratedProtocolMessageType('RectangleI', (_message.Message,), {
  'DESCRIPTOR' : _RECTANGLEI,
  '__module__' : 'lt_protobuf.light_header.rectanglei_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.RectangleI)
  })
_sym_db.RegisterMessage(RectangleI)


# @@protoc_insertion_point(module_scope)
