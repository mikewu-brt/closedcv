# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: camera_eeprom.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='camera_eeprom.proto',
  package='ltpb',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x13\x63\x61mera_eeprom.proto\x12\x04ltpb\"\xfa\x01\n\x0c\x43\x61meraEEPROM\x12\x36\n\reeprom_bundle\x18\x01 \x03(\x0b\x32\x1f.ltpb.CameraEEPROM.ModuleEEPROM\x1a\x87\x01\n\x0cModuleEEPROM\x12(\n\x04type\x18\x01 \x01(\x0e\x32\x1a.ltpb.CameraEEPROM.Project\x12\x10\n\x08sub_type\x18\x02 \x01(\r\x12\x13\n\x0bmodule_name\x18\x03 \x01(\t\x12\x11\n\tdata_size\x18\x04 \x01(\r\x12\x13\n\x0b\x64\x61ta_offset\x18\x05 \x01(\r\"(\n\x07Project\x12\x07\n\x03L16\x10\x00\x12\t\n\x05LIBRA\x10\x01\x12\t\n\x05ORION\x10\x02\x62\x06proto3'
)



_CAMERAEEPROM_PROJECT = _descriptor.EnumDescriptor(
  name='Project',
  full_name='ltpb.CameraEEPROM.Project',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='L16', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LIBRA', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ORION', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=240,
  serialized_end=280,
)
_sym_db.RegisterEnumDescriptor(_CAMERAEEPROM_PROJECT)


_CAMERAEEPROM_MODULEEEPROM = _descriptor.Descriptor(
  name='ModuleEEPROM',
  full_name='ltpb.CameraEEPROM.ModuleEEPROM',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='ltpb.CameraEEPROM.ModuleEEPROM.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sub_type', full_name='ltpb.CameraEEPROM.ModuleEEPROM.sub_type', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='module_name', full_name='ltpb.CameraEEPROM.ModuleEEPROM.module_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_size', full_name='ltpb.CameraEEPROM.ModuleEEPROM.data_size', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_offset', full_name='ltpb.CameraEEPROM.ModuleEEPROM.data_offset', index=4,
      number=5, type=13, cpp_type=3, label=1,
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
  serialized_start=103,
  serialized_end=238,
)

_CAMERAEEPROM = _descriptor.Descriptor(
  name='CameraEEPROM',
  full_name='ltpb.CameraEEPROM',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='eeprom_bundle', full_name='ltpb.CameraEEPROM.eeprom_bundle', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_CAMERAEEPROM_MODULEEEPROM, ],
  enum_types=[
    _CAMERAEEPROM_PROJECT,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=30,
  serialized_end=280,
)

_CAMERAEEPROM_MODULEEEPROM.fields_by_name['type'].enum_type = _CAMERAEEPROM_PROJECT
_CAMERAEEPROM_MODULEEEPROM.containing_type = _CAMERAEEPROM
_CAMERAEEPROM.fields_by_name['eeprom_bundle'].message_type = _CAMERAEEPROM_MODULEEEPROM
_CAMERAEEPROM_PROJECT.containing_type = _CAMERAEEPROM
DESCRIPTOR.message_types_by_name['CameraEEPROM'] = _CAMERAEEPROM
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

CameraEEPROM = _reflection.GeneratedProtocolMessageType('CameraEEPROM', (_message.Message,), {

  'ModuleEEPROM' : _reflection.GeneratedProtocolMessageType('ModuleEEPROM', (_message.Message,), {
    'DESCRIPTOR' : _CAMERAEEPROM_MODULEEEPROM,
    '__module__' : 'camera_eeprom_pb2'
    # @@protoc_insertion_point(class_scope:ltpb.CameraEEPROM.ModuleEEPROM)
    })
  ,
  'DESCRIPTOR' : _CAMERAEEPROM,
  '__module__' : 'camera_eeprom_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.CameraEEPROM)
  })
_sym_db.RegisterMessage(CameraEEPROM)
_sym_db.RegisterMessage(CameraEEPROM.ModuleEEPROM)


# @@protoc_insertion_point(module_scope)
