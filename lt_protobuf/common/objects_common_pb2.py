# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lt_protobuf/common/objects_common.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import wrappers_pb2 as google_dot_protobuf_dot_wrappers__pb2
from lt_protobuf.common import rectanglef_pb2 as lt__protobuf_dot_common_dot_rectanglef__pb2
from lt_protobuf.common import point2i_pb2 as lt__protobuf_dot_common_dot_point2i__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='lt_protobuf/common/objects_common.proto',
  package='ltpb.objects',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\'lt_protobuf/common/objects_common.proto\x12\x0cltpb.objects\x1a\x1egoogle/protobuf/wrappers.proto\x1a#lt_protobuf/common/rectanglef.proto\x1a lt_protobuf/common/point2i.proto\",\n\nObjectInfo\x12\x10\n\x08\x63lass_id\x18\x01 \x01(\x05\x12\x0c\n\x04name\x18\x02 \x01(\t\"\xa2\x01\n\x0f\x44\x65tectionResult\x12.\n\x0cobject_class\x18\x01 \x01(\x0b\x32\x18.ltpb.objects.ObjectInfo\x12\x17\n\x0f\x64\x65tection_score\x18\x02 \x01(\x02\x12\x13\n\x0b\x63lass_score\x18\x03 \x01(\x02\x12\x31\n\x0binstance_id\x18\x04 \x01(\x0b\x32\x1c.google.protobuf.UInt32Value\"4\n\x07ImageUC\x12\x1b\n\x04size\x18\x01 \x01(\x0b\x32\r.ltpb.Point2I\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\x0c\",\n\rNameValuePair\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02\"\xe7\x02\n\x06Metric\x12\x35\n\x0b\x65ngine_name\x18\x01 \x01(\x0e\x32 .ltpb.objects.Metric.DepthEngine\x12\x34\n\x0b\x65ngine_type\x18\x02 \x01(\x0e\x32\x1f.ltpb.objects.Metric.EngineType\x12\x36\n\x0cmetric_group\x18\x03 \x01(\x0e\x32 .ltpb.objects.Metric.MetricGroup\x12*\n\x05value\x18\x04 \x03(\x0b\x32\x1b.ltpb.objects.NameValuePair\"6\n\x0b\x44\x65pthEngine\x12\t\n\x05LIGHT\x10\x00\x12\x0c\n\x08LIGHT_ML\x10\x01\x12\x0e\n\nOPENCV_SGM\x10\x02\"+\n\nEngineType\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\x10\n\x0c\x45\x44GE_ALIGNED\x10\x01\"\'\n\x0bMetricGroup\x12\t\n\x05\x44\x45PTH\x10\x00\x12\r\n\tDISPARITY\x10\x01\"\xd2\x01\n\tReference\x12\x33\n\x04type\x18\x01 \x01(\x0e\x32%.ltpb.objects.Reference.ReferenceType\x12\r\n\x05\x64\x65pth\x18\x02 \x01(\x02\"\x80\x01\n\rReferenceType\x12\n\n\x06MANUAL\x10\x00\x12\r\n\tSYNTHETIC\x10\x01\x12\r\n\tHDMAP_3DM\x10\x02\x12\x19\n\x15LIDAR_VELODYNE_VLP32C\x10\x03\x12\x18\n\x14LIDAR_OUSTER_OS2_128\x10\x04\x12\x10\n\x0cGPS_GOMENTUM\x10\x05\"\xf7\x02\n\x0f\x41nnotatedObject\x12\x0c\n\x04name\x18\x01 \x01(\t\x12&\n\x0c\x62ounding_box\x18\x02 \x01(\x0b\x32\x10.ltpb.RectangleF\x12\x37\n\x10\x64\x65tection_result\x18\x03 \x01(\x0b\x32\x1d.ltpb.objects.DetectionResult\x12\x41\n\rdetector_type\x18\x04 \x01(\x0e\x32*.ltpb.objects.AnnotatedObject.DetectorType\x12#\n\x04mask\x18\x05 \x01(\x0b\x32\x15.ltpb.objects.ImageUC\x12%\n\x07metrics\x18\x06 \x03(\x0b\x32\x14.ltpb.objects.Metric\x12+\n\nreferences\x18\x07 \x03(\x0b\x32\x17.ltpb.objects.Reference\"9\n\x0c\x44\x65tectorType\x12\x12\n\x0eOBJECT_TRACKER\x10\x00\x12\t\n\x05HDMAP\x10\x01\x12\n\n\x06MANUAL\x10\x02\x62\x06proto3'
  ,
  dependencies=[google_dot_protobuf_dot_wrappers__pb2.DESCRIPTOR,lt__protobuf_dot_common_dot_rectanglef__pb2.DESCRIPTOR,lt__protobuf_dot_common_dot_point2i__pb2.DESCRIPTOR,])



_METRIC_DEPTHENGINE = _descriptor.EnumDescriptor(
  name='DepthEngine',
  full_name='ltpb.objects.Metric.DepthEngine',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='LIGHT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LIGHT_ML', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OPENCV_SGM', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=691,
  serialized_end=745,
)
_sym_db.RegisterEnumDescriptor(_METRIC_DEPTHENGINE)

_METRIC_ENGINETYPE = _descriptor.EnumDescriptor(
  name='EngineType',
  full_name='ltpb.objects.Metric.EngineType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DEFAULT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EDGE_ALIGNED', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=747,
  serialized_end=790,
)
_sym_db.RegisterEnumDescriptor(_METRIC_ENGINETYPE)

_METRIC_METRICGROUP = _descriptor.EnumDescriptor(
  name='MetricGroup',
  full_name='ltpb.objects.Metric.MetricGroup',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DEPTH', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DISPARITY', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=792,
  serialized_end=831,
)
_sym_db.RegisterEnumDescriptor(_METRIC_METRICGROUP)

_REFERENCE_REFERENCETYPE = _descriptor.EnumDescriptor(
  name='ReferenceType',
  full_name='ltpb.objects.Reference.ReferenceType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='MANUAL', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYNTHETIC', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HDMAP_3DM', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LIDAR_VELODYNE_VLP32C', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LIDAR_OUSTER_OS2_128', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GPS_GOMENTUM', index=5, number=5,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=916,
  serialized_end=1044,
)
_sym_db.RegisterEnumDescriptor(_REFERENCE_REFERENCETYPE)

_ANNOTATEDOBJECT_DETECTORTYPE = _descriptor.EnumDescriptor(
  name='DetectorType',
  full_name='ltpb.objects.AnnotatedObject.DetectorType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='OBJECT_TRACKER', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HDMAP', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MANUAL', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1365,
  serialized_end=1422,
)
_sym_db.RegisterEnumDescriptor(_ANNOTATEDOBJECT_DETECTORTYPE)


_OBJECTINFO = _descriptor.Descriptor(
  name='ObjectInfo',
  full_name='ltpb.objects.ObjectInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='class_id', full_name='ltpb.objects.ObjectInfo.class_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='ltpb.objects.ObjectInfo.name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=160,
  serialized_end=204,
)


_DETECTIONRESULT = _descriptor.Descriptor(
  name='DetectionResult',
  full_name='ltpb.objects.DetectionResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='object_class', full_name='ltpb.objects.DetectionResult.object_class', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='detection_score', full_name='ltpb.objects.DetectionResult.detection_score', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='class_score', full_name='ltpb.objects.DetectionResult.class_score', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='instance_id', full_name='ltpb.objects.DetectionResult.instance_id', index=3,
      number=4, type=11, cpp_type=10, label=1,
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
  serialized_start=207,
  serialized_end=369,
)


_IMAGEUC = _descriptor.Descriptor(
  name='ImageUC',
  full_name='ltpb.objects.ImageUC',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='size', full_name='ltpb.objects.ImageUC.size', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='ltpb.objects.ImageUC.data', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
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
  serialized_start=371,
  serialized_end=423,
)


_NAMEVALUEPAIR = _descriptor.Descriptor(
  name='NameValuePair',
  full_name='ltpb.objects.NameValuePair',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='ltpb.objects.NameValuePair.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='ltpb.objects.NameValuePair.value', index=1,
      number=2, type=2, cpp_type=6, label=1,
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
  serialized_start=425,
  serialized_end=469,
)


_METRIC = _descriptor.Descriptor(
  name='Metric',
  full_name='ltpb.objects.Metric',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='engine_name', full_name='ltpb.objects.Metric.engine_name', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='engine_type', full_name='ltpb.objects.Metric.engine_type', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='metric_group', full_name='ltpb.objects.Metric.metric_group', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='ltpb.objects.Metric.value', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _METRIC_DEPTHENGINE,
    _METRIC_ENGINETYPE,
    _METRIC_METRICGROUP,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=472,
  serialized_end=831,
)


_REFERENCE = _descriptor.Descriptor(
  name='Reference',
  full_name='ltpb.objects.Reference',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='ltpb.objects.Reference.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='depth', full_name='ltpb.objects.Reference.depth', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _REFERENCE_REFERENCETYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=834,
  serialized_end=1044,
)


_ANNOTATEDOBJECT = _descriptor.Descriptor(
  name='AnnotatedObject',
  full_name='ltpb.objects.AnnotatedObject',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='ltpb.objects.AnnotatedObject.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bounding_box', full_name='ltpb.objects.AnnotatedObject.bounding_box', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='detection_result', full_name='ltpb.objects.AnnotatedObject.detection_result', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='detector_type', full_name='ltpb.objects.AnnotatedObject.detector_type', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mask', full_name='ltpb.objects.AnnotatedObject.mask', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='metrics', full_name='ltpb.objects.AnnotatedObject.metrics', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='references', full_name='ltpb.objects.AnnotatedObject.references', index=6,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _ANNOTATEDOBJECT_DETECTORTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1047,
  serialized_end=1422,
)

_DETECTIONRESULT.fields_by_name['object_class'].message_type = _OBJECTINFO
_DETECTIONRESULT.fields_by_name['instance_id'].message_type = google_dot_protobuf_dot_wrappers__pb2._UINT32VALUE
_IMAGEUC.fields_by_name['size'].message_type = lt__protobuf_dot_common_dot_point2i__pb2._POINT2I
_METRIC.fields_by_name['engine_name'].enum_type = _METRIC_DEPTHENGINE
_METRIC.fields_by_name['engine_type'].enum_type = _METRIC_ENGINETYPE
_METRIC.fields_by_name['metric_group'].enum_type = _METRIC_METRICGROUP
_METRIC.fields_by_name['value'].message_type = _NAMEVALUEPAIR
_METRIC_DEPTHENGINE.containing_type = _METRIC
_METRIC_ENGINETYPE.containing_type = _METRIC
_METRIC_METRICGROUP.containing_type = _METRIC
_REFERENCE.fields_by_name['type'].enum_type = _REFERENCE_REFERENCETYPE
_REFERENCE_REFERENCETYPE.containing_type = _REFERENCE
_ANNOTATEDOBJECT.fields_by_name['bounding_box'].message_type = lt__protobuf_dot_common_dot_rectanglef__pb2._RECTANGLEF
_ANNOTATEDOBJECT.fields_by_name['detection_result'].message_type = _DETECTIONRESULT
_ANNOTATEDOBJECT.fields_by_name['detector_type'].enum_type = _ANNOTATEDOBJECT_DETECTORTYPE
_ANNOTATEDOBJECT.fields_by_name['mask'].message_type = _IMAGEUC
_ANNOTATEDOBJECT.fields_by_name['metrics'].message_type = _METRIC
_ANNOTATEDOBJECT.fields_by_name['references'].message_type = _REFERENCE
_ANNOTATEDOBJECT_DETECTORTYPE.containing_type = _ANNOTATEDOBJECT
DESCRIPTOR.message_types_by_name['ObjectInfo'] = _OBJECTINFO
DESCRIPTOR.message_types_by_name['DetectionResult'] = _DETECTIONRESULT
DESCRIPTOR.message_types_by_name['ImageUC'] = _IMAGEUC
DESCRIPTOR.message_types_by_name['NameValuePair'] = _NAMEVALUEPAIR
DESCRIPTOR.message_types_by_name['Metric'] = _METRIC
DESCRIPTOR.message_types_by_name['Reference'] = _REFERENCE
DESCRIPTOR.message_types_by_name['AnnotatedObject'] = _ANNOTATEDOBJECT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ObjectInfo = _reflection.GeneratedProtocolMessageType('ObjectInfo', (_message.Message,), {
  'DESCRIPTOR' : _OBJECTINFO,
  '__module__' : 'lt_protobuf.common.objects_common_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.objects.ObjectInfo)
  })
_sym_db.RegisterMessage(ObjectInfo)

DetectionResult = _reflection.GeneratedProtocolMessageType('DetectionResult', (_message.Message,), {
  'DESCRIPTOR' : _DETECTIONRESULT,
  '__module__' : 'lt_protobuf.common.objects_common_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.objects.DetectionResult)
  })
_sym_db.RegisterMessage(DetectionResult)

ImageUC = _reflection.GeneratedProtocolMessageType('ImageUC', (_message.Message,), {
  'DESCRIPTOR' : _IMAGEUC,
  '__module__' : 'lt_protobuf.common.objects_common_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.objects.ImageUC)
  })
_sym_db.RegisterMessage(ImageUC)

NameValuePair = _reflection.GeneratedProtocolMessageType('NameValuePair', (_message.Message,), {
  'DESCRIPTOR' : _NAMEVALUEPAIR,
  '__module__' : 'lt_protobuf.common.objects_common_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.objects.NameValuePair)
  })
_sym_db.RegisterMessage(NameValuePair)

Metric = _reflection.GeneratedProtocolMessageType('Metric', (_message.Message,), {
  'DESCRIPTOR' : _METRIC,
  '__module__' : 'lt_protobuf.common.objects_common_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.objects.Metric)
  })
_sym_db.RegisterMessage(Metric)

Reference = _reflection.GeneratedProtocolMessageType('Reference', (_message.Message,), {
  'DESCRIPTOR' : _REFERENCE,
  '__module__' : 'lt_protobuf.common.objects_common_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.objects.Reference)
  })
_sym_db.RegisterMessage(Reference)

AnnotatedObject = _reflection.GeneratedProtocolMessageType('AnnotatedObject', (_message.Message,), {
  'DESCRIPTOR' : _ANNOTATEDOBJECT,
  '__module__' : 'lt_protobuf.common.objects_common_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.objects.AnnotatedObject)
  })
_sym_db.RegisterMessage(AnnotatedObject)


# @@protoc_insertion_point(module_scope)
