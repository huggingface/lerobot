import pandas
import struct

# Robot exception message type
ROBOT_MESSAGE_RUNTIME_EXCEPTION = 10
ROBOT_MESSAGE_EXCEPTION = 6

# Robot exception data type
ROBOT_EXCEPTION_DATA_TYPE_NONE = 0
ROBOT_EXCEPTION_DATA_TYPE_UNSIGNED = 1
ROBOT_EXCEPTION_DATA_TYPE_SIGNED = 2
ROBOT_EXCEPTION_DATA_TYPE_FLOAT = 3
ROBOT_EXCEPTION_DATA_TYPE_HEX = 4
ROBOT_EXCEPTION_DATA_TYPE_STRING = 5
ROBOT_EXCEPTION_DATA_TYPE_JOINT = 6

class RobotHeader():
    __slots__ = ['type', 'size',]
    @staticmethod
    def unpack(buf):
        rmd = RobotHeader()
        (rmd.size, rmd.type) = struct.unpack_from('>iB', buf)
        return rmd

class RobotDataConfig():
    __slots__ = ['fmt', 'name']
    
    @staticmethod
    def __type_to_pack(vartype : str):
        if vartype == 'int' or vartype == 'int32_t':
            return 'i'
        elif vartype == 'int8_t' :
            return 'b'
        elif vartype == 'uint8_t':
            return 'B'
        elif vartype == 'uint64_t':
            return'Q'
        elif vartype == 'bool':
            return '?'
        elif vartype == 'double':
            return 'd'
        elif vartype == 'uint32_t':
            return 'I'
        elif vartype == 'float':
            return 'f'
        elif vartype == 'uint32_t':
            return 'I'
        else:
            raise("Unkow Type")

    @staticmethod
    def get_config(file, sheet):
        config = RobotDataConfig()
        excel = pandas.read_excel(file, sheet_name=sheet)
        config.fmt = '>'
        config.name = []
        is_foreach = False
        temp_fmt = ''
        temp_name = []
        elite_internel_count = 0
        for i in range(len(excel['type'])):
            if type(excel['name'][i]) == float:
                pass
            if type(excel['type'][i]) == str:
                excel['type'][i].replace(" ", "")
            if excel['type'][i] == 'bytes':
                config.fmt += 'B' * excel['bytes'][i]
                for j in range(excel['bytes'][i]):
                    config.name.append(excel['name'][i] + '_' +str(j) + '_' + str(elite_internel_count))
                    elite_internel_count += 1
            elif excel['type'][i] == 'foreach':
                is_foreach = True
            elif excel['type'][i] == 'end' and is_foreach == True:
                config.fmt += (temp_fmt * 6)
                for j in range(6):
                    for k in temp_name:
                        config.name.append(k + str(j))
                temp_fmt = ''
                temp_name = []
                is_foreach = False
            else:
                if is_foreach:
                    temp_name.append(excel['name'][i])
                    temp_fmt += RobotDataConfig.__type_to_pack(excel['type'][i])
                else:
                    config.name.append(excel['name'][i])
                    config.fmt += RobotDataConfig.__type_to_pack(excel['type'][i])
        return config

class RobotException():
    __slots__ = ['time_stamp', 'exception_source', 'exception_type', 'code', 'subcode', 'level', 'data', 'script_line', 'script_column', 'description']
    def __init__(self, time_stamp, exception_source, exception_type) -> None:
        self.time_stamp = time_stamp
        self.exception_source = exception_source
        self.exception_type = exception_type
    def set_robot_exception(self, code, subcode, level, data):
        self.code = code
        self.subcode = subcode
        self.level = level
        self.data = data
    def set_runtime_exception(self, script_line, script_column, description):
        self.script_line = script_line
        self.script_column = script_column
        self.description = description
    @staticmethod
    def unpack_exception(buffer:bytes):
        (pack_len, pack_type, timestamp, source, msg_type) = struct.unpack_from('>iBQBB', buffer)
        if msg_type == ROBOT_MESSAGE_RUNTIME_EXCEPTION:
            (pack_len, pack_type, timestamp, source, msg_type, script_line, script_column) = struct.unpack_from('>iBQBBii', buffer)
            exception = RobotException(timestamp, source, msg_type)
            description = buffer[struct.calcsize(">iBQBBii"):pack_len]
            description = description.decode("utf-8")
            exception.set_runtime_exception(script_line, script_column, description)
            return exception
        elif msg_type == ROBOT_MESSAGE_EXCEPTION:
            (pack_len, pack_type, timestamp, source, msg_type, code, subcode, level, data_type) = struct.unpack_from('>iBQBBiiii', buffer)
            if data_type == ROBOT_EXCEPTION_DATA_TYPE_NONE:
                (pack_len, pack_type, timestamp, source, msg_type, code, subcode, level, data_type, data) = struct.unpack_from('>iBQBBiiiiI', buffer)
            elif data_type == ROBOT_EXCEPTION_DATA_TYPE_UNSIGNED:
                (pack_len, pack_type, timestamp, source, msg_type, code, subcode, level, data_type, data) = struct.unpack_from('>iBQBBiiiiI', buffer)
            elif data_type == ROBOT_EXCEPTION_DATA_TYPE_SIGNED:
                (pack_len, pack_type, timestamp, source, msg_type, code, subcode, level, data_type, data) = struct.unpack_from('>iBQBBiiiii', buffer)
            elif data_type == ROBOT_EXCEPTION_DATA_TYPE_FLOAT:
                (pack_len, pack_type, timestamp, source, msg_type, code, subcode, level, data_type, data) = struct.unpack_from('>iBQBBiiiif', buffer)
            elif data_type == ROBOT_EXCEPTION_DATA_TYPE_HEX:
                (pack_len, pack_type, timestamp, source, msg_type, code, subcode, level, data_type, data) = struct.unpack_from('>iBQBBiiiiI', buffer)
            elif data_type == ROBOT_EXCEPTION_DATA_TYPE_STRING:
                data = buffer[struct.calcsize(">iBQBBiiii") : pack_len]
            elif data_type == ROBOT_EXCEPTION_DATA_TYPE_JOINT:
                (pack_len, pack_type, timestamp, source, msg_type, code, subcode, level, data_type, data) = struct.unpack_from('>iBQBBiiiiI', buffer)
            exception = RobotException(timestamp, source, msg_type)
            exception.set_robot_exception(code, subcode, level, data)
            return exception
            

class RobotData():
    @staticmethod
    def unpack(buf, config : RobotDataConfig):
        data = RobotData()
        unpack = struct.unpack_from(config.fmt, buf)
        for i in range(len(config.name)):
            name = config.name[i]
            data.__dict__[name] = unpack[i]
        return data