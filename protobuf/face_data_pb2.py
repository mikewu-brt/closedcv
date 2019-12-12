# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: face_data.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import camera_id_pb2 as camera__id__pb2
import rectanglei_pb2 as rectanglei__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='face_data.proto',
  package='ltpb',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x0f\x66\x61\x63\x65_data.proto\x12\x04ltpb\x1a\x0f\x63\x61mera_id.proto\x1a\x10rectanglei.proto\"\xb4\x01\n\x08\x46\x61\x63\x65\x44\x61ta\x12\x1a\n\x02id\x18\x01 \x01(\x0e\x32\x0e.ltpb.CameraID\x12\x13\n\x0b\x66rame_index\x18\x02 \x01(\r\x12 \n\x04rois\x18\x03 \x03(\x0b\x32\x12.ltpb.FaceData.ROI\x1aU\n\x03ROI\x12\x0c\n\x02id\x18\x01 \x01(\rH\x00\x12\x1d\n\x03roi\x18\x02 \x01(\x0b\x32\x10.ltpb.RectangleI\x12\x12\n\nconfidence\x18\x03 \x01(\x02\x42\r\n\x0boptional_idb\x06proto3'
  ,
  dependencies=[camera__id__pb2.DESCRIPTOR,rectanglei__pb2.DESCRIPTOR,])




_FACEDATA_ROI = _descriptor.Descriptor(
  name='ROI',
  full_name='ltpb.FaceData.ROI',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='ltpb.FaceData.ROI.id', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='roi', full_name='ltpb.FaceData.ROI.roi', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='confidence', full_name='ltpb.FaceData.ROI.confidence', index=2,
      number=3, type=2, cpp_type=6, label=1,
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
    _descriptor.OneofDescriptor(
      name='optional_id', full_name='ltpb.FaceData.ROI.optional_id',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=156,
  serialized_end=241,
)

_FACEDATA = _descriptor.Descriptor(
  name='FaceData',
  full_name='ltpb.FaceData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='ltpb.FaceData.id', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frame_index', full_name='ltpb.FaceData.frame_index', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rois', full_name='ltpb.FaceData.rois', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_FACEDATA_ROI, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=61,
  serialized_end=241,
)

_FACEDATA_ROI.fields_by_name['roi'].message_type = rectanglei__pb2._RECTANGLEI
_FACEDATA_ROI.containing_type = _FACEDATA
_FACEDATA_ROI.oneofs_by_name['optional_id'].fields.append(
  _FACEDATA_ROI.fields_by_name['id'])
_FACEDATA_ROI.fields_by_name['id'].containing_oneof = _FACEDATA_ROI.oneofs_by_name['optional_id']
_FACEDATA.fields_by_name['id'].enum_type = camera__id__pb2._CAMERAID
_FACEDATA.fields_by_name['rois'].message_type = _FACEDATA_ROI
DESCRIPTOR.message_types_by_name['FaceData'] = _FACEDATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FaceData = _reflection.GeneratedProtocolMessageType('FaceData', (_message.Message,), {

  'ROI' : _reflection.GeneratedProtocolMessageType('ROI', (_message.Message,), {
    'DESCRIPTOR' : _FACEDATA_ROI,
    '__module__' : 'face_data_pb2'
    # @@protoc_insertion_point(class_scope:ltpb.FaceData.ROI)
    })
  ,
  'DESCRIPTOR' : _FACEDATA,
  '__module__' : 'face_data_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.FaceData)
  })
_sym_db.RegisterMessage(FaceData)
_sym_db.RegisterMessage(FaceData.ROI)


# @@protoc_insertion_point(module_scope)
