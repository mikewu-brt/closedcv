# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lt_protobuf/light_header/flash_calibration.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='lt_protobuf/light_header/flash_calibration.proto',
  package='ltpb',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n0lt_protobuf/light_header/flash_calibration.proto\x12\x04ltpb\"\x9e\x01\n\x10\x46lashCalibration\x12\x13\n\x0bledcool_lux\x18\x01 \x01(\x02\x12\x1a\n\x12ledcool_max_lumens\x18\x02 \x01(\x02\x12\x13\n\x0bledcool_cct\x18\x03 \x01(\x02\x12\x13\n\x0bledwarm_lux\x18\x04 \x01(\x02\x12\x1a\n\x12ledwarm_max_lumens\x18\x05 \x01(\x02\x12\x13\n\x0bledwarm_cct\x18\x06 \x01(\x02\x62\x06proto3'
)




_FLASHCALIBRATION = _descriptor.Descriptor(
  name='FlashCalibration',
  full_name='ltpb.FlashCalibration',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ledcool_lux', full_name='ltpb.FlashCalibration.ledcool_lux', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ledcool_max_lumens', full_name='ltpb.FlashCalibration.ledcool_max_lumens', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ledcool_cct', full_name='ltpb.FlashCalibration.ledcool_cct', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ledwarm_lux', full_name='ltpb.FlashCalibration.ledwarm_lux', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ledwarm_max_lumens', full_name='ltpb.FlashCalibration.ledwarm_max_lumens', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ledwarm_cct', full_name='ltpb.FlashCalibration.ledwarm_cct', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=59,
  serialized_end=217,
)

DESCRIPTOR.message_types_by_name['FlashCalibration'] = _FLASHCALIBRATION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FlashCalibration = _reflection.GeneratedProtocolMessageType('FlashCalibration', (_message.Message,), {
  'DESCRIPTOR' : _FLASHCALIBRATION,
  '__module__' : 'lt_protobuf.light_header.flash_calibration_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.FlashCalibration)
  })
_sym_db.RegisterMessage(FlashCalibration)


# @@protoc_insertion_point(module_scope)