# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: sac.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'sac.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\tsac.proto\x12\x1blerobot.common.api.grpc.sac\"^\n\nModelState\x12\x42\n\x0etransfer_state\x18\x01 \x01(\x0e\x32*.lerobot.common.api.grpc.sac.TransferState\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\x0c\"\"\n\x0cModelRequest\x12\x12\n\nmodel_name\x18\x01 \x01(\t*`\n\rTransferState\x12\x14\n\x10TRANSFER_UNKNOWN\x10\x00\x12\x12\n\x0eTRANSFER_BEGIN\x10\x01\x12\x13\n\x0fTRANSFER_MIDDLE\x10\x02\x12\x10\n\x0cTRANSFER_END\x10\x03\x32\xdb\x01\n\rTensorService\x12j\n\x12StreamModelUpdates\x12).lerobot.common.api.grpc.sac.ModelRequest\x1a\'.lerobot.common.api.grpc.sac.ModelState0\x01\x12^\n\x08GetModel\x12).lerobot.common.api.grpc.sac.ModelRequest\x1a\'.lerobot.common.api.grpc.sac.ModelStateb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'sac_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_TRANSFERSTATE']._serialized_start=174
  _globals['_TRANSFERSTATE']._serialized_end=270
  _globals['_MODELSTATE']._serialized_start=42
  _globals['_MODELSTATE']._serialized_end=136
  _globals['_MODELREQUEST']._serialized_start=138
  _globals['_MODELREQUEST']._serialized_end=172
  _globals['_TENSORSERVICE']._serialized_start=273
  _globals['_TENSORSERVICE']._serialized_end=492
# @@protoc_insertion_point(module_scope)
