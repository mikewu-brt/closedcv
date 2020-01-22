# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: geometric_calibration.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import mirror_system_pb2 as mirror__system__pb2
import matrix3x3f_pb2 as matrix3x3f__pb2
import point3f_pb2 as point3f__pb2
import distortion_pb2 as distortion__pb2
import range2f_pb2 as range2f__pb2
import device_temp_pb2 as device__temp__pb2
import point2f_pb2 as point2f__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='geometric_calibration.proto',
  package='ltpb',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x1bgeometric_calibration.proto\x12\x04ltpb\x1a\x13mirror_system.proto\x1a\x10matrix3x3f.proto\x1a\rpoint3f.proto\x1a\x10\x64istortion.proto\x1a\rrange2f.proto\x1a\x11\x64\x65vice_temp.proto\x1a\rpoint2f.proto\"\xe8\n\n\x14GeometricCalibration\x12:\n\x0bmirror_type\x18\x01 \x01(\x0e\x32%.ltpb.GeometricCalibration.MirrorType\x12P\n\x15per_focus_calibration\x18\x02 \x03(\x0b\x32\x31.ltpb.GeometricCalibration.CalibrationFocusBundle\x12$\n\ndistortion\x18\x03 \x01(\x0b\x32\x10.ltpb.Distortion\x12+\n\x14lens_hall_code_range\x18\x04 \x01(\x0b\x32\r.ltpb.Range2F\x12+\n\x14\x66ocus_distance_range\x18\x05 \x01(\x0b\x32\r.ltpb.Range2F\x12Z\n\x1c\x61ngle_optical_center_mapping\x18\x06 \x01(\x0b\x32\x34.ltpb.GeometricCalibration.AngleOpticalCenterMapping\x1a@\n\nIntrinsics\x12\x1f\n\x05k_mat\x18\x01 \x01(\x0b\x32\x10.ltpb.Matrix3x3F\x12\x11\n\trms_error\x18\x02 \x01(\x02\x1a\xb8\x03\n\nExtrinsics\x12H\n\tcanonical\x18\x01 \x01(\x0b\x32\x35.ltpb.GeometricCalibration.Extrinsics.CanonicalFormat\x12R\n\x0fmoveable_mirror\x18\x02 \x01(\x0b\x32\x39.ltpb.GeometricCalibration.Extrinsics.MovableMirrorFormat\x1a\x8b\x01\n\x0f\x43\x61nonicalFormat\x12\"\n\x08rotation\x18\x01 \x01(\x0b\x32\x10.ltpb.Matrix3x3F\x12\"\n\x0btranslation\x18\x02 \x01(\x0b\x32\r.ltpb.Point3F\x12\x14\n\x0cstereo_error\x18\x03 \x01(\x02\x12\x1a\n\x12reprojection_error\x18\x04 \x01(\x02\x1a~\n\x13MovableMirrorFormat\x12)\n\rmirror_system\x18\x01 \x01(\x0b\x32\x12.ltpb.MirrorSystem\x12<\n\x17mirror_actuator_mapping\x18\x02 \x01(\x0b\x32\x1b.ltpb.MirrorActuatorMapping\x1a\x99\x02\n\x16\x43\x61librationFocusBundle\x12\x16\n\x0e\x66ocus_distance\x18\x01 \x01(\x02\x12\x39\n\nintrinsics\x18\x02 \x01(\x0b\x32%.ltpb.GeometricCalibration.Intrinsics\x12\x39\n\nextrinsics\x18\x03 \x01(\x0b\x32%.ltpb.GeometricCalibration.Extrinsics\x12\x13\n\x0bsensor_temp\x18\x04 \x01(\x11\x12%\n\x0b\x64\x65vice_temp\x18\x05 \x01(\x0b\x32\x10.ltpb.DeviceTemp\x12\x19\n\x0f\x66ocus_hall_code\x18\x06 \x01(\x02H\x00\x42\x1a\n\x18optional_focus_hall_code\x1a\x9c\x01\n\x19\x41ngleOpticalCenterMapping\x12#\n\x0c\x63\x65nter_start\x18\x01 \x01(\x0b\x32\r.ltpb.Point2F\x12!\n\ncenter_end\x18\x02 \x01(\x0b\x32\r.ltpb.Point2F\x12\x14\n\x0c\x61ngle_offset\x18\x03 \x01(\x02\x12\x0f\n\x07t_scale\x18\x04 \x01(\x02\x12\x10\n\x08t_offset\x18\x05 \x01(\x02\".\n\nMirrorType\x12\x08\n\x04NONE\x10\x00\x12\t\n\x05GLUED\x10\x01\x12\x0b\n\x07MOVABLE\x10\x02\x62\x06proto3'
  ,
  dependencies=[mirror__system__pb2.DESCRIPTOR,matrix3x3f__pb2.DESCRIPTOR,point3f__pb2.DESCRIPTOR,distortion__pb2.DESCRIPTOR,range2f__pb2.DESCRIPTOR,device__temp__pb2.DESCRIPTOR,point2f__pb2.DESCRIPTOR,])



_GEOMETRICCALIBRATION_MIRRORTYPE = _descriptor.EnumDescriptor(
  name='MirrorType',
  full_name='ltpb.GeometricCalibration.MirrorType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NONE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GLUED', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MOVABLE', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1497,
  serialized_end=1543,
)
_sym_db.RegisterEnumDescriptor(_GEOMETRICCALIBRATION_MIRRORTYPE)


_GEOMETRICCALIBRATION_INTRINSICS = _descriptor.Descriptor(
  name='Intrinsics',
  full_name='ltpb.GeometricCalibration.Intrinsics',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='k_mat', full_name='ltpb.GeometricCalibration.Intrinsics.k_mat', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rms_error', full_name='ltpb.GeometricCalibration.Intrinsics.rms_error', index=1,
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
  serialized_start=545,
  serialized_end=609,
)

_GEOMETRICCALIBRATION_EXTRINSICS_CANONICALFORMAT = _descriptor.Descriptor(
  name='CanonicalFormat',
  full_name='ltpb.GeometricCalibration.Extrinsics.CanonicalFormat',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='rotation', full_name='ltpb.GeometricCalibration.Extrinsics.CanonicalFormat.rotation', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='translation', full_name='ltpb.GeometricCalibration.Extrinsics.CanonicalFormat.translation', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stereo_error', full_name='ltpb.GeometricCalibration.Extrinsics.CanonicalFormat.stereo_error', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reprojection_error', full_name='ltpb.GeometricCalibration.Extrinsics.CanonicalFormat.reprojection_error', index=3,
      number=4, type=2, cpp_type=6, label=1,
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
  serialized_start=785,
  serialized_end=924,
)

_GEOMETRICCALIBRATION_EXTRINSICS_MOVABLEMIRRORFORMAT = _descriptor.Descriptor(
  name='MovableMirrorFormat',
  full_name='ltpb.GeometricCalibration.Extrinsics.MovableMirrorFormat',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='mirror_system', full_name='ltpb.GeometricCalibration.Extrinsics.MovableMirrorFormat.mirror_system', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mirror_actuator_mapping', full_name='ltpb.GeometricCalibration.Extrinsics.MovableMirrorFormat.mirror_actuator_mapping', index=1,
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
  serialized_start=926,
  serialized_end=1052,
)

_GEOMETRICCALIBRATION_EXTRINSICS = _descriptor.Descriptor(
  name='Extrinsics',
  full_name='ltpb.GeometricCalibration.Extrinsics',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='canonical', full_name='ltpb.GeometricCalibration.Extrinsics.canonical', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='moveable_mirror', full_name='ltpb.GeometricCalibration.Extrinsics.moveable_mirror', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_GEOMETRICCALIBRATION_EXTRINSICS_CANONICALFORMAT, _GEOMETRICCALIBRATION_EXTRINSICS_MOVABLEMIRRORFORMAT, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=612,
  serialized_end=1052,
)

_GEOMETRICCALIBRATION_CALIBRATIONFOCUSBUNDLE = _descriptor.Descriptor(
  name='CalibrationFocusBundle',
  full_name='ltpb.GeometricCalibration.CalibrationFocusBundle',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='focus_distance', full_name='ltpb.GeometricCalibration.CalibrationFocusBundle.focus_distance', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='intrinsics', full_name='ltpb.GeometricCalibration.CalibrationFocusBundle.intrinsics', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='extrinsics', full_name='ltpb.GeometricCalibration.CalibrationFocusBundle.extrinsics', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sensor_temp', full_name='ltpb.GeometricCalibration.CalibrationFocusBundle.sensor_temp', index=3,
      number=4, type=17, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='device_temp', full_name='ltpb.GeometricCalibration.CalibrationFocusBundle.device_temp', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='focus_hall_code', full_name='ltpb.GeometricCalibration.CalibrationFocusBundle.focus_hall_code', index=5,
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
    _descriptor.OneofDescriptor(
      name='optional_focus_hall_code', full_name='ltpb.GeometricCalibration.CalibrationFocusBundle.optional_focus_hall_code',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=1055,
  serialized_end=1336,
)

_GEOMETRICCALIBRATION_ANGLEOPTICALCENTERMAPPING = _descriptor.Descriptor(
  name='AngleOpticalCenterMapping',
  full_name='ltpb.GeometricCalibration.AngleOpticalCenterMapping',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='center_start', full_name='ltpb.GeometricCalibration.AngleOpticalCenterMapping.center_start', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='center_end', full_name='ltpb.GeometricCalibration.AngleOpticalCenterMapping.center_end', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='angle_offset', full_name='ltpb.GeometricCalibration.AngleOpticalCenterMapping.angle_offset', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='t_scale', full_name='ltpb.GeometricCalibration.AngleOpticalCenterMapping.t_scale', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='t_offset', full_name='ltpb.GeometricCalibration.AngleOpticalCenterMapping.t_offset', index=4,
      number=5, type=2, cpp_type=6, label=1,
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
  serialized_start=1339,
  serialized_end=1495,
)

_GEOMETRICCALIBRATION = _descriptor.Descriptor(
  name='GeometricCalibration',
  full_name='ltpb.GeometricCalibration',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='mirror_type', full_name='ltpb.GeometricCalibration.mirror_type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='per_focus_calibration', full_name='ltpb.GeometricCalibration.per_focus_calibration', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='distortion', full_name='ltpb.GeometricCalibration.distortion', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lens_hall_code_range', full_name='ltpb.GeometricCalibration.lens_hall_code_range', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='focus_distance_range', full_name='ltpb.GeometricCalibration.focus_distance_range', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='angle_optical_center_mapping', full_name='ltpb.GeometricCalibration.angle_optical_center_mapping', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_GEOMETRICCALIBRATION_INTRINSICS, _GEOMETRICCALIBRATION_EXTRINSICS, _GEOMETRICCALIBRATION_CALIBRATIONFOCUSBUNDLE, _GEOMETRICCALIBRATION_ANGLEOPTICALCENTERMAPPING, ],
  enum_types=[
    _GEOMETRICCALIBRATION_MIRRORTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=159,
  serialized_end=1543,
)

_GEOMETRICCALIBRATION_INTRINSICS.fields_by_name['k_mat'].message_type = matrix3x3f__pb2._MATRIX3X3F
_GEOMETRICCALIBRATION_INTRINSICS.containing_type = _GEOMETRICCALIBRATION
_GEOMETRICCALIBRATION_EXTRINSICS_CANONICALFORMAT.fields_by_name['rotation'].message_type = matrix3x3f__pb2._MATRIX3X3F
_GEOMETRICCALIBRATION_EXTRINSICS_CANONICALFORMAT.fields_by_name['translation'].message_type = point3f__pb2._POINT3F
_GEOMETRICCALIBRATION_EXTRINSICS_CANONICALFORMAT.containing_type = _GEOMETRICCALIBRATION_EXTRINSICS
_GEOMETRICCALIBRATION_EXTRINSICS_MOVABLEMIRRORFORMAT.fields_by_name['mirror_system'].message_type = mirror__system__pb2._MIRRORSYSTEM
_GEOMETRICCALIBRATION_EXTRINSICS_MOVABLEMIRRORFORMAT.fields_by_name['mirror_actuator_mapping'].message_type = mirror__system__pb2._MIRRORACTUATORMAPPING
_GEOMETRICCALIBRATION_EXTRINSICS_MOVABLEMIRRORFORMAT.containing_type = _GEOMETRICCALIBRATION_EXTRINSICS
_GEOMETRICCALIBRATION_EXTRINSICS.fields_by_name['canonical'].message_type = _GEOMETRICCALIBRATION_EXTRINSICS_CANONICALFORMAT
_GEOMETRICCALIBRATION_EXTRINSICS.fields_by_name['moveable_mirror'].message_type = _GEOMETRICCALIBRATION_EXTRINSICS_MOVABLEMIRRORFORMAT
_GEOMETRICCALIBRATION_EXTRINSICS.containing_type = _GEOMETRICCALIBRATION
_GEOMETRICCALIBRATION_CALIBRATIONFOCUSBUNDLE.fields_by_name['intrinsics'].message_type = _GEOMETRICCALIBRATION_INTRINSICS
_GEOMETRICCALIBRATION_CALIBRATIONFOCUSBUNDLE.fields_by_name['extrinsics'].message_type = _GEOMETRICCALIBRATION_EXTRINSICS
_GEOMETRICCALIBRATION_CALIBRATIONFOCUSBUNDLE.fields_by_name['device_temp'].message_type = device__temp__pb2._DEVICETEMP
_GEOMETRICCALIBRATION_CALIBRATIONFOCUSBUNDLE.containing_type = _GEOMETRICCALIBRATION
_GEOMETRICCALIBRATION_CALIBRATIONFOCUSBUNDLE.oneofs_by_name['optional_focus_hall_code'].fields.append(
  _GEOMETRICCALIBRATION_CALIBRATIONFOCUSBUNDLE.fields_by_name['focus_hall_code'])
_GEOMETRICCALIBRATION_CALIBRATIONFOCUSBUNDLE.fields_by_name['focus_hall_code'].containing_oneof = _GEOMETRICCALIBRATION_CALIBRATIONFOCUSBUNDLE.oneofs_by_name['optional_focus_hall_code']
_GEOMETRICCALIBRATION_ANGLEOPTICALCENTERMAPPING.fields_by_name['center_start'].message_type = point2f__pb2._POINT2F
_GEOMETRICCALIBRATION_ANGLEOPTICALCENTERMAPPING.fields_by_name['center_end'].message_type = point2f__pb2._POINT2F
_GEOMETRICCALIBRATION_ANGLEOPTICALCENTERMAPPING.containing_type = _GEOMETRICCALIBRATION
_GEOMETRICCALIBRATION.fields_by_name['mirror_type'].enum_type = _GEOMETRICCALIBRATION_MIRRORTYPE
_GEOMETRICCALIBRATION.fields_by_name['per_focus_calibration'].message_type = _GEOMETRICCALIBRATION_CALIBRATIONFOCUSBUNDLE
_GEOMETRICCALIBRATION.fields_by_name['distortion'].message_type = distortion__pb2._DISTORTION
_GEOMETRICCALIBRATION.fields_by_name['lens_hall_code_range'].message_type = range2f__pb2._RANGE2F
_GEOMETRICCALIBRATION.fields_by_name['focus_distance_range'].message_type = range2f__pb2._RANGE2F
_GEOMETRICCALIBRATION.fields_by_name['angle_optical_center_mapping'].message_type = _GEOMETRICCALIBRATION_ANGLEOPTICALCENTERMAPPING
_GEOMETRICCALIBRATION_MIRRORTYPE.containing_type = _GEOMETRICCALIBRATION
DESCRIPTOR.message_types_by_name['GeometricCalibration'] = _GEOMETRICCALIBRATION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GeometricCalibration = _reflection.GeneratedProtocolMessageType('GeometricCalibration', (_message.Message,), {

  'Intrinsics' : _reflection.GeneratedProtocolMessageType('Intrinsics', (_message.Message,), {
    'DESCRIPTOR' : _GEOMETRICCALIBRATION_INTRINSICS,
    '__module__' : 'geometric_calibration_pb2'
    # @@protoc_insertion_point(class_scope:ltpb.GeometricCalibration.Intrinsics)
    })
  ,

  'Extrinsics' : _reflection.GeneratedProtocolMessageType('Extrinsics', (_message.Message,), {

    'CanonicalFormat' : _reflection.GeneratedProtocolMessageType('CanonicalFormat', (_message.Message,), {
      'DESCRIPTOR' : _GEOMETRICCALIBRATION_EXTRINSICS_CANONICALFORMAT,
      '__module__' : 'geometric_calibration_pb2'
      # @@protoc_insertion_point(class_scope:ltpb.GeometricCalibration.Extrinsics.CanonicalFormat)
      })
    ,

    'MovableMirrorFormat' : _reflection.GeneratedProtocolMessageType('MovableMirrorFormat', (_message.Message,), {
      'DESCRIPTOR' : _GEOMETRICCALIBRATION_EXTRINSICS_MOVABLEMIRRORFORMAT,
      '__module__' : 'geometric_calibration_pb2'
      # @@protoc_insertion_point(class_scope:ltpb.GeometricCalibration.Extrinsics.MovableMirrorFormat)
      })
    ,
    'DESCRIPTOR' : _GEOMETRICCALIBRATION_EXTRINSICS,
    '__module__' : 'geometric_calibration_pb2'
    # @@protoc_insertion_point(class_scope:ltpb.GeometricCalibration.Extrinsics)
    })
  ,

  'CalibrationFocusBundle' : _reflection.GeneratedProtocolMessageType('CalibrationFocusBundle', (_message.Message,), {
    'DESCRIPTOR' : _GEOMETRICCALIBRATION_CALIBRATIONFOCUSBUNDLE,
    '__module__' : 'geometric_calibration_pb2'
    # @@protoc_insertion_point(class_scope:ltpb.GeometricCalibration.CalibrationFocusBundle)
    })
  ,

  'AngleOpticalCenterMapping' : _reflection.GeneratedProtocolMessageType('AngleOpticalCenterMapping', (_message.Message,), {
    'DESCRIPTOR' : _GEOMETRICCALIBRATION_ANGLEOPTICALCENTERMAPPING,
    '__module__' : 'geometric_calibration_pb2'
    # @@protoc_insertion_point(class_scope:ltpb.GeometricCalibration.AngleOpticalCenterMapping)
    })
  ,
  'DESCRIPTOR' : _GEOMETRICCALIBRATION,
  '__module__' : 'geometric_calibration_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.GeometricCalibration)
  })
_sym_db.RegisterMessage(GeometricCalibration)
_sym_db.RegisterMessage(GeometricCalibration.Intrinsics)
_sym_db.RegisterMessage(GeometricCalibration.Extrinsics)
_sym_db.RegisterMessage(GeometricCalibration.Extrinsics.CanonicalFormat)
_sym_db.RegisterMessage(GeometricCalibration.Extrinsics.MovableMirrorFormat)
_sym_db.RegisterMessage(GeometricCalibration.CalibrationFocusBundle)
_sym_db.RegisterMessage(GeometricCalibration.AngleOpticalCenterMapping)


# @@protoc_insertion_point(module_scope)