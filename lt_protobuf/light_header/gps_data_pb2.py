# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lt_protobuf/light_header/gps_data.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='lt_protobuf/light_header/gps_data.proto',
  package='ltpb',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\'lt_protobuf/light_header/gps_data.proto\x12\x04ltpb\"\xb2\x0b\n\x07GPSData\x12\x39\n\x11processing_method\x18\x01 \x01(\x0e\x32\x1e.ltpb.GPSData.ProcessingMethod\x12.\n\x0cgps_fix_mode\x18\x02 \x01(\x0e\x32\x18.ltpb.GPSData.GpsFixMode\x12\x1a\n\x12satellites_in_view\x18\x03 \x01(\r\x12\"\n\x05track\x18\x04 \x01(\x0b\x32\x13.ltpb.GPSData.Track\x12&\n\x07heading\x18\x05 \x01(\x0b\x32\x15.ltpb.GPSData.Heading\x12(\n\x08\x61ltitude\x18\x06 \x01(\x0b\x32\x16.ltpb.GPSData.Altitude\x12,\n\x04time\x18\x07 \x01(\x0b\x32\x1e.ltpb.GPSData.ValueUncertainty\x12\x30\n\x08latitude\x18\x08 \x01(\x0b\x32\x1e.ltpb.GPSData.ValueUncertainty\x12\x31\n\tlongitude\x18\t \x01(\x0b\x32\x1e.ltpb.GPSData.ValueUncertainty\x12-\n\x05speed\x18\n \x01(\x0b\x32\x1e.ltpb.GPSData.ValueUncertainty\x12-\n\x05\x63limb\x18\x0b \x01(\x0b\x32\x1e.ltpb.GPSData.ValueUncertainty\x12.\n\x03\x64op\x18\x0c \x01(\x0b\x32!.ltpb.GPSData.DilutionOfPrecision\x1a\x36\n\x10ValueUncertainty\x12\r\n\x05value\x18\x01 \x01(\x01\x12\x13\n\x0buncertainty\x18\x02 \x01(\x01\x1a\x61\n\x05Track\x12-\n\x05value\x18\x01 \x01(\x0b\x32\x1e.ltpb.GPSData.ValueUncertainty\x12)\n\x03ref\x18\x02 \x01(\x0e\x32\x1c.ltpb.GPSData.ReferenceNorth\x1a\x63\n\x07Heading\x12-\n\x05value\x18\x01 \x01(\x0b\x32\x1e.ltpb.GPSData.ValueUncertainty\x12)\n\x03ref\x18\x02 \x01(\x0e\x32\x1c.ltpb.GPSData.ReferenceNorth\x1ag\n\x08\x41ltitude\x12-\n\x05value\x18\x01 \x01(\x0b\x32\x1e.ltpb.GPSData.ValueUncertainty\x12,\n\x03ref\x18\x02 \x01(\x0e\x32\x1f.ltpb.GPSData.ReferenceAltitude\x1aw\n\x13\x44ilutionOfPrecision\x12\x0c\n\x04xdop\x18\x01 \x01(\x01\x12\x0c\n\x04ydop\x18\x02 \x01(\x01\x12\x0c\n\x04hdop\x18\x04 \x01(\x01\x12\x0c\n\x04vdop\x18\x05 \x01(\x01\x12\x0c\n\x04gdop\x18\x07 \x01(\x01\x12\x0c\n\x04pdop\x18\x03 \x01(\x01\x12\x0c\n\x04tdop\x18\x06 \x01(\x01\"b\n\nGpsFixMode\x12\x15\n\x11\x46IX_MODE_NOT_SEEN\x10\x00\x12\x13\n\x0f\x46IX_MODE_NO_FIX\x10\x01\x12\x12\n\x0e\x46IX_MODE_TWO_D\x10\x02\x12\x14\n\x10\x46IX_MODE_THREE_D\x10\x03\"H\n\x0eReferenceNorth\x12\x1c\n\x18REFERENCE_NORTH_MAGNETIC\x10\x00\x12\x18\n\x14REFERENCE_NORTH_TRUE\x10\x01\"5\n\x11ReferenceAltitude\x12 \n\x1cREFERENCE_ALTITUDE_SEA_LEVEL\x10\x00\"\xc1\x01\n\x10ProcessingMethod\x12\x1d\n\x19PROCESSING_METHOD_UNKNOWN\x10\x00\x12\x19\n\x15PROCESSING_METHOD_GPS\x10\x01\x12\x1c\n\x18PROCESSING_METHOD_CELLID\x10\x02\x12\x1a\n\x16PROCESSING_METHOD_WLAN\x10\x03\x12\x1c\n\x18PROCESSING_METHOD_MANUAL\x10\x04\x12\x1b\n\x17PROCESSING_METHOD_FUSED\x10\x05\x62\x06proto3'
)



_GPSDATA_GPSFIXMODE = _descriptor.EnumDescriptor(
  name='GpsFixMode',
  full_name='ltpb.GPSData.GpsFixMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='FIX_MODE_NOT_SEEN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FIX_MODE_NO_FIX', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FIX_MODE_TWO_D', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FIX_MODE_THREE_D', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1085,
  serialized_end=1183,
)
_sym_db.RegisterEnumDescriptor(_GPSDATA_GPSFIXMODE)

_GPSDATA_REFERENCENORTH = _descriptor.EnumDescriptor(
  name='ReferenceNorth',
  full_name='ltpb.GPSData.ReferenceNorth',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='REFERENCE_NORTH_MAGNETIC', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REFERENCE_NORTH_TRUE', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1185,
  serialized_end=1257,
)
_sym_db.RegisterEnumDescriptor(_GPSDATA_REFERENCENORTH)

_GPSDATA_REFERENCEALTITUDE = _descriptor.EnumDescriptor(
  name='ReferenceAltitude',
  full_name='ltpb.GPSData.ReferenceAltitude',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='REFERENCE_ALTITUDE_SEA_LEVEL', index=0, number=0,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1259,
  serialized_end=1312,
)
_sym_db.RegisterEnumDescriptor(_GPSDATA_REFERENCEALTITUDE)

_GPSDATA_PROCESSINGMETHOD = _descriptor.EnumDescriptor(
  name='ProcessingMethod',
  full_name='ltpb.GPSData.ProcessingMethod',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='PROCESSING_METHOD_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PROCESSING_METHOD_GPS', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PROCESSING_METHOD_CELLID', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PROCESSING_METHOD_WLAN', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PROCESSING_METHOD_MANUAL', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PROCESSING_METHOD_FUSED', index=5, number=5,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1315,
  serialized_end=1508,
)
_sym_db.RegisterEnumDescriptor(_GPSDATA_PROCESSINGMETHOD)


_GPSDATA_VALUEUNCERTAINTY = _descriptor.Descriptor(
  name='ValueUncertainty',
  full_name='ltpb.GPSData.ValueUncertainty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='ltpb.GPSData.ValueUncertainty.value', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='uncertainty', full_name='ltpb.GPSData.ValueUncertainty.uncertainty', index=1,
      number=2, type=1, cpp_type=5, label=1,
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
  serialized_start=603,
  serialized_end=657,
)

_GPSDATA_TRACK = _descriptor.Descriptor(
  name='Track',
  full_name='ltpb.GPSData.Track',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='ltpb.GPSData.Track.value', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ref', full_name='ltpb.GPSData.Track.ref', index=1,
      number=2, type=14, cpp_type=8, label=1,
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
  serialized_start=659,
  serialized_end=756,
)

_GPSDATA_HEADING = _descriptor.Descriptor(
  name='Heading',
  full_name='ltpb.GPSData.Heading',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='ltpb.GPSData.Heading.value', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ref', full_name='ltpb.GPSData.Heading.ref', index=1,
      number=2, type=14, cpp_type=8, label=1,
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
  serialized_start=758,
  serialized_end=857,
)

_GPSDATA_ALTITUDE = _descriptor.Descriptor(
  name='Altitude',
  full_name='ltpb.GPSData.Altitude',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='ltpb.GPSData.Altitude.value', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ref', full_name='ltpb.GPSData.Altitude.ref', index=1,
      number=2, type=14, cpp_type=8, label=1,
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
  serialized_start=859,
  serialized_end=962,
)

_GPSDATA_DILUTIONOFPRECISION = _descriptor.Descriptor(
  name='DilutionOfPrecision',
  full_name='ltpb.GPSData.DilutionOfPrecision',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='xdop', full_name='ltpb.GPSData.DilutionOfPrecision.xdop', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ydop', full_name='ltpb.GPSData.DilutionOfPrecision.ydop', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hdop', full_name='ltpb.GPSData.DilutionOfPrecision.hdop', index=2,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='vdop', full_name='ltpb.GPSData.DilutionOfPrecision.vdop', index=3,
      number=5, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gdop', full_name='ltpb.GPSData.DilutionOfPrecision.gdop', index=4,
      number=7, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pdop', full_name='ltpb.GPSData.DilutionOfPrecision.pdop', index=5,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tdop', full_name='ltpb.GPSData.DilutionOfPrecision.tdop', index=6,
      number=6, type=1, cpp_type=5, label=1,
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
  serialized_start=964,
  serialized_end=1083,
)

_GPSDATA = _descriptor.Descriptor(
  name='GPSData',
  full_name='ltpb.GPSData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='processing_method', full_name='ltpb.GPSData.processing_method', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gps_fix_mode', full_name='ltpb.GPSData.gps_fix_mode', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='satellites_in_view', full_name='ltpb.GPSData.satellites_in_view', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='track', full_name='ltpb.GPSData.track', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='heading', full_name='ltpb.GPSData.heading', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='altitude', full_name='ltpb.GPSData.altitude', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='time', full_name='ltpb.GPSData.time', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='latitude', full_name='ltpb.GPSData.latitude', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='longitude', full_name='ltpb.GPSData.longitude', index=8,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='speed', full_name='ltpb.GPSData.speed', index=9,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='climb', full_name='ltpb.GPSData.climb', index=10,
      number=11, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dop', full_name='ltpb.GPSData.dop', index=11,
      number=12, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_GPSDATA_VALUEUNCERTAINTY, _GPSDATA_TRACK, _GPSDATA_HEADING, _GPSDATA_ALTITUDE, _GPSDATA_DILUTIONOFPRECISION, ],
  enum_types=[
    _GPSDATA_GPSFIXMODE,
    _GPSDATA_REFERENCENORTH,
    _GPSDATA_REFERENCEALTITUDE,
    _GPSDATA_PROCESSINGMETHOD,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=50,
  serialized_end=1508,
)

_GPSDATA_VALUEUNCERTAINTY.containing_type = _GPSDATA
_GPSDATA_TRACK.fields_by_name['value'].message_type = _GPSDATA_VALUEUNCERTAINTY
_GPSDATA_TRACK.fields_by_name['ref'].enum_type = _GPSDATA_REFERENCENORTH
_GPSDATA_TRACK.containing_type = _GPSDATA
_GPSDATA_HEADING.fields_by_name['value'].message_type = _GPSDATA_VALUEUNCERTAINTY
_GPSDATA_HEADING.fields_by_name['ref'].enum_type = _GPSDATA_REFERENCENORTH
_GPSDATA_HEADING.containing_type = _GPSDATA
_GPSDATA_ALTITUDE.fields_by_name['value'].message_type = _GPSDATA_VALUEUNCERTAINTY
_GPSDATA_ALTITUDE.fields_by_name['ref'].enum_type = _GPSDATA_REFERENCEALTITUDE
_GPSDATA_ALTITUDE.containing_type = _GPSDATA
_GPSDATA_DILUTIONOFPRECISION.containing_type = _GPSDATA
_GPSDATA.fields_by_name['processing_method'].enum_type = _GPSDATA_PROCESSINGMETHOD
_GPSDATA.fields_by_name['gps_fix_mode'].enum_type = _GPSDATA_GPSFIXMODE
_GPSDATA.fields_by_name['track'].message_type = _GPSDATA_TRACK
_GPSDATA.fields_by_name['heading'].message_type = _GPSDATA_HEADING
_GPSDATA.fields_by_name['altitude'].message_type = _GPSDATA_ALTITUDE
_GPSDATA.fields_by_name['time'].message_type = _GPSDATA_VALUEUNCERTAINTY
_GPSDATA.fields_by_name['latitude'].message_type = _GPSDATA_VALUEUNCERTAINTY
_GPSDATA.fields_by_name['longitude'].message_type = _GPSDATA_VALUEUNCERTAINTY
_GPSDATA.fields_by_name['speed'].message_type = _GPSDATA_VALUEUNCERTAINTY
_GPSDATA.fields_by_name['climb'].message_type = _GPSDATA_VALUEUNCERTAINTY
_GPSDATA.fields_by_name['dop'].message_type = _GPSDATA_DILUTIONOFPRECISION
_GPSDATA_GPSFIXMODE.containing_type = _GPSDATA
_GPSDATA_REFERENCENORTH.containing_type = _GPSDATA
_GPSDATA_REFERENCEALTITUDE.containing_type = _GPSDATA
_GPSDATA_PROCESSINGMETHOD.containing_type = _GPSDATA
DESCRIPTOR.message_types_by_name['GPSData'] = _GPSDATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GPSData = _reflection.GeneratedProtocolMessageType('GPSData', (_message.Message,), {

  'ValueUncertainty' : _reflection.GeneratedProtocolMessageType('ValueUncertainty', (_message.Message,), {
    'DESCRIPTOR' : _GPSDATA_VALUEUNCERTAINTY,
    '__module__' : 'lt_protobuf.light_header.gps_data_pb2'
    # @@protoc_insertion_point(class_scope:ltpb.GPSData.ValueUncertainty)
    })
  ,

  'Track' : _reflection.GeneratedProtocolMessageType('Track', (_message.Message,), {
    'DESCRIPTOR' : _GPSDATA_TRACK,
    '__module__' : 'lt_protobuf.light_header.gps_data_pb2'
    # @@protoc_insertion_point(class_scope:ltpb.GPSData.Track)
    })
  ,

  'Heading' : _reflection.GeneratedProtocolMessageType('Heading', (_message.Message,), {
    'DESCRIPTOR' : _GPSDATA_HEADING,
    '__module__' : 'lt_protobuf.light_header.gps_data_pb2'
    # @@protoc_insertion_point(class_scope:ltpb.GPSData.Heading)
    })
  ,

  'Altitude' : _reflection.GeneratedProtocolMessageType('Altitude', (_message.Message,), {
    'DESCRIPTOR' : _GPSDATA_ALTITUDE,
    '__module__' : 'lt_protobuf.light_header.gps_data_pb2'
    # @@protoc_insertion_point(class_scope:ltpb.GPSData.Altitude)
    })
  ,

  'DilutionOfPrecision' : _reflection.GeneratedProtocolMessageType('DilutionOfPrecision', (_message.Message,), {
    'DESCRIPTOR' : _GPSDATA_DILUTIONOFPRECISION,
    '__module__' : 'lt_protobuf.light_header.gps_data_pb2'
    # @@protoc_insertion_point(class_scope:ltpb.GPSData.DilutionOfPrecision)
    })
  ,
  'DESCRIPTOR' : _GPSDATA,
  '__module__' : 'lt_protobuf.light_header.gps_data_pb2'
  # @@protoc_insertion_point(class_scope:ltpb.GPSData)
  })
_sym_db.RegisterMessage(GPSData)
_sym_db.RegisterMessage(GPSData.ValueUncertainty)
_sym_db.RegisterMessage(GPSData.Track)
_sym_db.RegisterMessage(GPSData.Heading)
_sym_db.RegisterMessage(GPSData.Altitude)
_sym_db.RegisterMessage(GPSData.DilutionOfPrecision)


# @@protoc_insertion_point(module_scope)