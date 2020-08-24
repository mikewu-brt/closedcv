# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lt_protobuf/light_header/hw_info.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from lt_protobuf.light_header import sensor_type_pb2 as lt__protobuf_dot_light__header_dot_sensor__type__pb2
from lt_protobuf.light_header import camera_id_pb2 as lt__protobuf_dot_light__header_dot_camera__id__pb2
from lt_protobuf.light_header import point4f_pb2 as lt__protobuf_dot_light__header_dot_point4f__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='lt_protobuf/light_header/hw_info.proto',
  package='ltpb',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n&lt_protobuf/light_header/hw_info.proto\x12\x04ltpb\x1a*lt_protobuf/light_header/sensor_type.proto\x1a(lt_protobuf/light_header/camera_id.proto\x1a&lt_protobuf/light_header/point4f.proto\"\xac\x08\n\x12\x43\x61meraModuleHwInfo\x12\x1a\n\x02id\x18\x01 \x01(\x0e\x32\x0e.ltpb.CameraID\x12 \n\x06sensor\x18\x02 \x01(\x0e\x32\x10.ltpb.SensorType\x12/\n\x04lens\x18\x03 \x01(\x0e\x32!.ltpb.CameraModuleHwInfo.LensType\x12\x44\n\x0fmirror_actuator\x18\x04 \x01(\x0e\x32+.ltpb.CameraModuleHwInfo.MirrorActuatorType\x12\x33\n\x06mirror\x18\x05 \x01(\x0e\x32#.ltpb.CameraModuleHwInfo.MirrorType\x12\x17\n\x0f\x66ocal_length_mm\x18\x06 \x01(\x02\x12 \n\x16\x66ocal_length_temp_degc\x18\x07 \x01(\x02H\x00\x12(\n\x1e\x66ocal_length_slope_mm_per_degc\x18\x08 \x01(\x02H\x01\x12\x15\n\rpixel_size_mm\x18\t \x01(\x02\x12\x13\n\x0bsensor_uuid\x18\n \x01(\x04\x12\x15\n\rserial_number\x18\x0b \x01(\t\x12\x14\n\x0cmanufacturer\x18\x0c \x01(\t\"\x9c\x02\n\x08LensType\x12\x10\n\x0cLENS_UNKNOWN\x10\x00\x12\x0f\n\x0bLENS_SHOWIN\x10\x01\x12\x0f\n\x0bLENS_LARGAN\x10\x02\x12\x0e\n\nLENS_SUNNY\x10\x03\x12\x11\n\rLENS_KANTATSU\x10\x04\x12\x16\n\x12LENS_OFILM_60068A1\x10\x05\x12\x15\n\x11LENS_SEMCO_202SVA\x10\x06\x12\x15\n\x11LENS_SEMCO_162SVA\x10\x07\x12\x17\n\x13LENS_LARGAN_70013A1\x10\x08\x12\x1b\n\x17LENS_COMPUTAR_V2528_MPY\x10\t\x12\x1b\n\x17LENS_COMPUTAR_V3528_MPY\x10\n\x12 \n\x1cLENS_IMAGING_SOURCE_TPL_1220\x10\x0b\"^\n\x10LensActuatorType\x12\x19\n\x15LENS_ACTUATOR_UNKNOWN\x10\x00\x12\x18\n\x14LENS_ACTUATOR_SHICOH\x10\x01\x12\x15\n\x11LENS_ACTUATOR_PZT\x10\x02\"J\n\x12MirrorActuatorType\x12\x1b\n\x17MIRROR_ACTUATOR_UNKNOWN\x10\x00\x12\x17\n\x13MIRROR_ACTUATOR_PZT\x10\x01\"U\n\nMirrorType\x12\x12\n\x0eMIRROR_UNKNOWN\x10\x00\x12\x19\n\x15MIRROR_DIELECTRIC_SNX\x10\x01\x12\x18\n\x14MIRROR_SILVER_ZUISHO\x10\x02\x42!\n\x1foptional_focal_length_temp_degcB)\n\'optional_focal_length_slope_mm_per_degc\"\x9a\x01\n\x0fImuModuleHwInfo\x12\x11\n\tdevice_id\x18\x01 \x01(\x04\x12\x13\n\x0blocation_id\x18\x02 \x01(\r\x12\x14\n\x0cmanufacturer\x18\x03 \x01(\t\x12\x12\n\nproduct_id\x18\x04 \x01(\t\x12\x12\n\nfw_version\x18\x05 \x01(\t\x12!\n\nquaternion\x18\x06 \x01(\x0b\x32\r.ltpb.Point4F\"\xbf\x02\n\x06HwInfo\x12(\n\x06\x63\x61mera\x18\x01 \x03(\x0b\x32\x18.ltpb.CameraModuleHwInfo\x12%\n\x05\x66lash\x18\x02 \x01(\x0e\x32\x16.ltpb.HwInfo.FlashType\x12!\n\x03tof\x18\x03 \x01(\x0e\x32\x14.ltpb.HwInfo.ToFType\x12\x14\n\x0cmanufacturer\x18\x05 \x01(\t\x12\"\n\x03imu\x18\x06 \x03(\x0b\x32\x15.ltpb.ImuModuleHwInfo\x12\x0f\n\x07version\x18\x08 \x01(\t\"6\n\tFlashType\x12\x11\n\rFLASH_UNKNOWN\x10\x00\x12\x16\n\x12\x46LASH_OSRAM_CBLPM1\x10\x01\"2\n\x07ToFType\x12\x0f\n\x0bTOF_UNKNOWN\x10\x00\x12\x16\n\x12TOF_STMICRO_VL53L0\x10\x01J\x04\x08\x04\x10\x05J\x04\x08\x07\x10\x08\x62\x06proto3')
  ,
  dependencies=[lt__protobuf_dot_light__header_dot_sensor__type__pb2.DESCRIPTOR,lt__protobuf_dot_light__header_dot_camera__id__pb2.DESCRIPTOR,lt__protobuf_dot_light__header_dot_point4f__pb2.DESCRIPTOR,])



_CAMERAMODULEHWINFO_LENSTYPE = _descriptor.EnumDescriptor(
  name='LensType',
  full_name='ltpb.CameraModuleHwInfo.LensType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='LENS_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LENS_SHOWIN', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LENS_LARGAN', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LENS_SUNNY', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LENS_KANTATSU', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LENS_OFILM_60068A1', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LENS_SEMCO_202SVA', index=6, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LENS_SEMCO_162SVA', index=7, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LENS_LARGAN_70013A1', index=8, number=8,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LENS_COMPUTAR_V2528_MPY', index=9, number=9,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LENS_COMPUTAR_V3528_MPY', index=10, number=10,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LENS_IMAGING_SOURCE_TPL_1220', index=11, number=11,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=622,
  serialized_end=906,
)
_sym_db.RegisterEnumDescriptor(_CAMERAMODULEHWINFO_LENSTYPE)

_CAMERAMODULEHWINFO_LENSACTUATORTYPE = _descriptor.EnumDescriptor(
  name='LensActuatorType',
  full_name='ltpb.CameraModuleHwInfo.LensActuatorType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='LENS_ACTUATOR_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LENS_ACTUATOR_SHICOH', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LENS_ACTUATOR_PZT', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=908,
  serialized_end=1002,
)
_sym_db.RegisterEnumDescriptor(_CAMERAMODULEHWINFO_LENSACTUATORTYPE)

_CAMERAMODULEHWINFO_MIRRORACTUATORTYPE = _descriptor.EnumDescriptor(
  name='MirrorActuatorType',
  full_name='ltpb.CameraModuleHwInfo.MirrorActuatorType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='MIRROR_ACTUATOR_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MIRROR_ACTUATOR_PZT', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1004,
  serialized_end=1078,
)
_sym_db.RegisterEnumDescriptor(_CAMERAMODULEHWINFO_MIRRORACTUATORTYPE)

_CAMERAMODULEHWINFO_MIRRORTYPE = _descriptor.EnumDescriptor(
  name='MirrorType',
  full_name='ltpb.CameraModuleHwInfo.MirrorType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='MIRROR_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MIRROR_DIELECTRIC_SNX', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MIRROR_SILVER_ZUISHO', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1080,
  serialized_end=1165,
)
_sym_db.RegisterEnumDescriptor(_CAMERAMODULEHWINFO_MIRRORTYPE)

_HWINFO_FLASHTYPE = _descriptor.EnumDescriptor(
  name='FlashType',
  full_name='ltpb.HwInfo.FlashType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='FLASH_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FLASH_OSRAM_CBLPM1', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1604,
  serialized_end=1658,
)
_sym_db.RegisterEnumDescriptor(_HWINFO_FLASHTYPE)

_HWINFO_TOFTYPE = _descriptor.EnumDescriptor(
  name='ToFType',
  full_name='ltpb.HwInfo.ToFType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TOF_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TOF_STMICRO_VL53L0', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1660,
  serialized_end=1710,
)
_sym_db.RegisterEnumDescriptor(_HWINFO_TOFTYPE)


_CAMERAMODULEHWINFO = _descriptor.Descriptor(
  name='CameraModuleHwInfo',
  full_name='ltpb.CameraModuleHwInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='ltpb.CameraModuleHwInfo.id', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sensor', full_name='ltpb.CameraModuleHwInfo.sensor', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lens', full_name='ltpb.CameraModuleHwInfo.lens', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mirror_actuator', full_name='ltpb.CameraModuleHwInfo.mirror_actuator', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mirror', full_name='ltpb.CameraModuleHwInfo.mirror', index=4,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='focal_length_mm', full_name='ltpb.CameraModuleHwInfo.focal_length_mm', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='focal_length_temp_degc', full_name='ltpb.CameraModuleHwInfo.focal_length_temp_degc', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='focal_length_slope_mm_per_degc', full_name='ltpb.CameraModuleHwInfo.focal_length_slope_mm_per_degc', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pixel_size_mm', full_name='ltpb.CameraModuleHwInfo.pixel_size_mm', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sensor_uuid', full_name='ltpb.CameraModuleHwInfo.sensor_uuid', index=9,
      number=10, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='serial_number', full_name='ltpb.CameraModuleHwInfo.serial_number', index=10,
      number=11, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='manufacturer', full_name='ltpb.CameraModuleHwInfo.manufacturer', index=11,
      number=12, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _CAMERAMODULEHWINFO_LENSTYPE,
    _CAMERAMODULEHWINFO_LENSACTUATORTYPE,
    _CAMERAMODULEHWINFO_MIRRORACTUATORTYPE,
    _CAMERAMODULEHWINFO_MIRRORTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='optional_focal_length_temp_degc', full_name='ltpb.CameraModuleHwInfo.optional_focal_length_temp_degc',
      index=0, containing_type=None, fields=[]),
    _descriptor.OneofDescriptor(
      name='optional_focal_length_slope_mm_per_degc', full_name='ltpb.CameraModuleHwInfo.optional_focal_length_slope_mm_per_degc',
      index=1, containing_type=None, fields=[]),
  ],
  serialized_start=175,
  serialized_end=1243,
)


_IMUMODULEHWINFO = _descriptor.Descriptor(
  name='ImuModuleHwInfo',
  full_name='ltpb.ImuModuleHwInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='device_id', full_name='ltpb.ImuModuleHwInfo.device_id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='location_id', full_name='ltpb.ImuModuleHwInfo.location_id', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='manufacturer', full_name='ltpb.ImuModuleHwInfo.manufacturer', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='product_id', full_name='ltpb.ImuModuleHwInfo.product_id', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='fw_version', full_name='ltpb.ImuModuleHwInfo.fw_version', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='quaternion', full_name='ltpb.ImuModuleHwInfo.quaternion', index=5,
      number=6, type=11, cpp_type=10, label=1,
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
  serialized_start=1246,
  serialized_end=1400,
)


_HWINFO = _descriptor.Descriptor(
  name='HwInfo',
  full_name='ltpb.HwInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='camera', full_name='ltpb.HwInfo.camera', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='flash', full_name='ltpb.HwInfo.flash', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tof', full_name='ltpb.HwInfo.tof', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='manufacturer', full_name='ltpb.HwInfo.manufacturer', index=3,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='imu', full_name='ltpb.HwInfo.imu', index=4,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='version', full_name='ltpb.HwInfo.version', index=5,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _HWINFO_FLASHTYPE,
    _HWINFO_TOFTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1403,
  serialized_end=1722,
)

_CAMERAMODULEHWINFO.fields_by_name['id'].enum_type = lt__protobuf_dot_light__header_dot_camera__id__pb2._CAMERAID
_CAMERAMODULEHWINFO.fields_by_name['sensor'].enum_type = lt__protobuf_dot_light__header_dot_sensor__type__pb2._SENSORTYPE
_CAMERAMODULEHWINFO.fields_by_name['lens'].enum_type = _CAMERAMODULEHWINFO_LENSTYPE
_CAMERAMODULEHWINFO.fields_by_name['mirror_actuator'].enum_type = _CAMERAMODULEHWINFO_MIRRORACTUATORTYPE
_CAMERAMODULEHWINFO.fields_by_name['mirror'].enum_type = _CAMERAMODULEHWINFO_MIRRORTYPE
_CAMERAMODULEHWINFO_LENSTYPE.containing_type = _CAMERAMODULEHWINFO
_CAMERAMODULEHWINFO_LENSACTUATORTYPE.containing_type = _CAMERAMODULEHWINFO
_CAMERAMODULEHWINFO_MIRRORACTUATORTYPE.containing_type = _CAMERAMODULEHWINFO
_CAMERAMODULEHWINFO_MIRRORTYPE.containing_type = _CAMERAMODULEHWINFO
_CAMERAMODULEHWINFO.oneofs_by_name['optional_focal_length_temp_degc'].fields.append(
  _CAMERAMODULEHWINFO.fields_by_name['focal_length_temp_degc'])
_CAMERAMODULEHWINFO.fields_by_name['focal_length_temp_degc'].containing_oneof = _CAMERAMODULEHWINFO.oneofs_by_name['optional_focal_length_temp_degc']
_CAMERAMODULEHWINFO.oneofs_by_name['optional_focal_length_slope_mm_per_degc'].fields.append(
  _CAMERAMODULEHWINFO.fields_by_name['focal_length_slope_mm_per_degc'])
_CAMERAMODULEHWINFO.fields_by_name['focal_length_slope_mm_per_degc'].containing_oneof = _CAMERAMODULEHWINFO.oneofs_by_name['optional_focal_length_slope_mm_per_degc']
_IMUMODULEHWINFO.fields_by_name['quaternion'].message_type = lt__protobuf_dot_light__header_dot_point4f__pb2._POINT4F
_HWINFO.fields_by_name['camera'].message_type = _CAMERAMODULEHWINFO
_HWINFO.fields_by_name['flash'].enum_type = _HWINFO_FLASHTYPE
_HWINFO.fields_by_name['tof'].enum_type = _HWINFO_TOFTYPE
_HWINFO.fields_by_name['imu'].message_type = _IMUMODULEHWINFO
_HWINFO_FLASHTYPE.containing_type = _HWINFO
_HWINFO_TOFTYPE.containing_type = _HWINFO
DESCRIPTOR.message_types_by_name['CameraModuleHwInfo'] = _CAMERAMODULEHWINFO
DESCRIPTOR.message_types_by_name['ImuModuleHwInfo'] = _IMUMODULEHWINFO
DESCRIPTOR.message_types_by_name['HwInfo'] = _HWINFO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

CameraModuleHwInfo = _reflection.GeneratedProtocolMessageType('CameraModuleHwInfo', (_message.Message,), {
  'DESCRIPTOR' : _CAMERAMODULEHWINFO,
  '__module__' : 'lt_protobuf.light_header.hw_info_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.CameraModuleHwInfo)
  })
_sym_db.RegisterMessage(CameraModuleHwInfo)

ImuModuleHwInfo = _reflection.GeneratedProtocolMessageType('ImuModuleHwInfo', (_message.Message,), {
  'DESCRIPTOR' : _IMUMODULEHWINFO,
  '__module__' : 'lt_protobuf.light_header.hw_info_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.ImuModuleHwInfo)
  })
_sym_db.RegisterMessage(ImuModuleHwInfo)

HwInfo = _reflection.GeneratedProtocolMessageType('HwInfo', (_message.Message,), {
  'DESCRIPTOR' : _HWINFO,
  '__module__' : 'lt_protobuf.light_header.hw_info_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.HwInfo)
  })
_sym_db.RegisterMessage(HwInfo)


# @@protoc_insertion_point(module_scope)
