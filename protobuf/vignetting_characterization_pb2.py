# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: vignetting_characterization.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import matrix4x4f_pb2 as matrix4x4f__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='vignetting_characterization.proto',
  package='ltpb',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n!vignetting_characterization.proto\x12\x04ltpb\x1a\x10matrix4x4f.proto\"\x81\x04\n\x1aVignettingCharacterization\x12\x42\n\tcrosstalk\x18\x01 \x01(\x0b\x32/.ltpb.VignettingCharacterization.CrosstalkModel\x12J\n\nvignetting\x18\x02 \x03(\x0b\x32\x36.ltpb.VignettingCharacterization.MirrorVignettingModel\x12\x1b\n\x13relative_brightness\x18\x03 \x01(\x02\x12\x16\n\x0elens_hall_code\x18\x04 \x01(\x05\x1ah\n\x0e\x43rosstalkModel\x12\r\n\x05width\x18\x01 \x01(\r\x12\x0e\n\x06height\x18\x02 \x01(\r\x12\x1e\n\x04\x64\x61ta\x18\x03 \x03(\x0b\x32\x10.ltpb.Matrix4x4F\x12\x17\n\x0b\x64\x61ta_packed\x18\x04 \x03(\x02\x42\x02\x10\x01\x1a\x42\n\x0fVignettingModel\x12\r\n\x05width\x18\x01 \x01(\r\x12\x0e\n\x06height\x18\x02 \x01(\r\x12\x10\n\x04\x64\x61ta\x18\x03 \x03(\x02\x42\x02\x10\x01\x1ap\n\x15MirrorVignettingModel\x12\x11\n\thall_code\x18\x01 \x01(\x05\x12\x44\n\nvignetting\x18\x02 \x01(\x0b\x32\x30.ltpb.VignettingCharacterization.VignettingModelb\x06proto3'
  ,
  dependencies=[matrix4x4f__pb2.DESCRIPTOR,])




_VIGNETTINGCHARACTERIZATION_CROSSTALKMODEL = _descriptor.Descriptor(
  name='CrosstalkModel',
  full_name='ltpb.VignettingCharacterization.CrosstalkModel',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='width', full_name='ltpb.VignettingCharacterization.CrosstalkModel.width', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='ltpb.VignettingCharacterization.CrosstalkModel.height', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='ltpb.VignettingCharacterization.CrosstalkModel.data', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_packed', full_name='ltpb.VignettingCharacterization.CrosstalkModel.data_packed', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\020\001', file=DESCRIPTOR),
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
  serialized_start=289,
  serialized_end=393,
)

_VIGNETTINGCHARACTERIZATION_VIGNETTINGMODEL = _descriptor.Descriptor(
  name='VignettingModel',
  full_name='ltpb.VignettingCharacterization.VignettingModel',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='width', full_name='ltpb.VignettingCharacterization.VignettingModel.width', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='ltpb.VignettingCharacterization.VignettingModel.height', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='ltpb.VignettingCharacterization.VignettingModel.data', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\020\001', file=DESCRIPTOR),
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
  serialized_start=395,
  serialized_end=461,
)

_VIGNETTINGCHARACTERIZATION_MIRRORVIGNETTINGMODEL = _descriptor.Descriptor(
  name='MirrorVignettingModel',
  full_name='ltpb.VignettingCharacterization.MirrorVignettingModel',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='hall_code', full_name='ltpb.VignettingCharacterization.MirrorVignettingModel.hall_code', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='vignetting', full_name='ltpb.VignettingCharacterization.MirrorVignettingModel.vignetting', index=1,
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
  serialized_start=463,
  serialized_end=575,
)

_VIGNETTINGCHARACTERIZATION = _descriptor.Descriptor(
  name='VignettingCharacterization',
  full_name='ltpb.VignettingCharacterization',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='crosstalk', full_name='ltpb.VignettingCharacterization.crosstalk', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='vignetting', full_name='ltpb.VignettingCharacterization.vignetting', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='relative_brightness', full_name='ltpb.VignettingCharacterization.relative_brightness', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lens_hall_code', full_name='ltpb.VignettingCharacterization.lens_hall_code', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_VIGNETTINGCHARACTERIZATION_CROSSTALKMODEL, _VIGNETTINGCHARACTERIZATION_VIGNETTINGMODEL, _VIGNETTINGCHARACTERIZATION_MIRRORVIGNETTINGMODEL, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=62,
  serialized_end=575,
)

_VIGNETTINGCHARACTERIZATION_CROSSTALKMODEL.fields_by_name['data'].message_type = matrix4x4f__pb2._MATRIX4X4F
_VIGNETTINGCHARACTERIZATION_CROSSTALKMODEL.containing_type = _VIGNETTINGCHARACTERIZATION
_VIGNETTINGCHARACTERIZATION_VIGNETTINGMODEL.containing_type = _VIGNETTINGCHARACTERIZATION
_VIGNETTINGCHARACTERIZATION_MIRRORVIGNETTINGMODEL.fields_by_name['vignetting'].message_type = _VIGNETTINGCHARACTERIZATION_VIGNETTINGMODEL
_VIGNETTINGCHARACTERIZATION_MIRRORVIGNETTINGMODEL.containing_type = _VIGNETTINGCHARACTERIZATION
_VIGNETTINGCHARACTERIZATION.fields_by_name['crosstalk'].message_type = _VIGNETTINGCHARACTERIZATION_CROSSTALKMODEL
_VIGNETTINGCHARACTERIZATION.fields_by_name['vignetting'].message_type = _VIGNETTINGCHARACTERIZATION_MIRRORVIGNETTINGMODEL
DESCRIPTOR.message_types_by_name['VignettingCharacterization'] = _VIGNETTINGCHARACTERIZATION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

VignettingCharacterization = _reflection.GeneratedProtocolMessageType('VignettingCharacterization', (_message.Message,), {

  'CrosstalkModel' : _reflection.GeneratedProtocolMessageType('CrosstalkModel', (_message.Message,), {
    'DESCRIPTOR' : _VIGNETTINGCHARACTERIZATION_CROSSTALKMODEL,
    '__module__' : 'vignetting_characterization_pb2'
    # @@protoc_insertion_point(class_scope:ltpb.VignettingCharacterization.CrosstalkModel)
    })
  ,

  'VignettingModel' : _reflection.GeneratedProtocolMessageType('VignettingModel', (_message.Message,), {
    'DESCRIPTOR' : _VIGNETTINGCHARACTERIZATION_VIGNETTINGMODEL,
    '__module__' : 'vignetting_characterization_pb2'
    # @@protoc_insertion_point(class_scope:ltpb.VignettingCharacterization.VignettingModel)
    })
  ,

  'MirrorVignettingModel' : _reflection.GeneratedProtocolMessageType('MirrorVignettingModel', (_message.Message,), {
    'DESCRIPTOR' : _VIGNETTINGCHARACTERIZATION_MIRRORVIGNETTINGMODEL,
    '__module__' : 'vignetting_characterization_pb2'
    # @@protoc_insertion_point(class_scope:ltpb.VignettingCharacterization.MirrorVignettingModel)
    })
  ,
  'DESCRIPTOR' : _VIGNETTINGCHARACTERIZATION,
  '__module__' : 'vignetting_characterization_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.VignettingCharacterization)
  })
_sym_db.RegisterMessage(VignettingCharacterization)
_sym_db.RegisterMessage(VignettingCharacterization.CrosstalkModel)
_sym_db.RegisterMessage(VignettingCharacterization.VignettingModel)
_sym_db.RegisterMessage(VignettingCharacterization.MirrorVignettingModel)


_VIGNETTINGCHARACTERIZATION_CROSSTALKMODEL.fields_by_name['data_packed']._options = None
_VIGNETTINGCHARACTERIZATION_VIGNETTINGMODEL.fields_by_name['data']._options = None
# @@protoc_insertion_point(module_scope)
